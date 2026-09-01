# MIT License
#
# Copyright (c) 2026 @CedrickArmel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Assign-split stage: folder of extract manifests -> one stable split manifest.

Behavior-anchored through the public ``radiologist.etl.assign_splits`` API
only — no internal helper is imported directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_record(path: str, filename: str, **overrides: object) -> object:
    from radiologist.etl import ManifestRecord

    base: dict[str, object] = dict(
        manifest_id="extract-0000000000000001",
        path=path,
        filename=filename,
        label="NORMAL",
        split="",
        stats={"haralick_contrast": 1.0},
    )
    base.update(overrides)
    return ManifestRecord(**base)  # type: ignore[arg-type]


def _write_manifest(directory: Path, name: str, records: list) -> str:
    from radiologist.etl import JsonlWriter

    dest = str(directory / name)
    JsonlWriter().write(records, dest, storage_options=None)
    return dest


class TestAssignSplitsBasics:
    def test_one_input_manifest_produces_one_split_manifest_with_same_records(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        records = [
            _make_record("/d/a.png", "a.png"),
            _make_record("/d/b.png", "b.png"),
        ]
        _write_manifest(manifests_dir, "extract-0001.jsonl", records)

        result = assign_splits(str(manifests_dir), str(dest_dir))

        written = records_reader(result.split_manifest_path)
        assert {r.path for r in written} == {"/d/a.png", "/d/b.png"}
        assert all(r.split for r in written)

    def test_three_input_manifests_yield_the_union_and_report_the_source_count(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir, "extract-0001.jsonl", [_make_record("/d/a.png", "a.png")]
        )
        _write_manifest(
            manifests_dir, "extract-0002.jsonl", [_make_record("/d/b.png", "b.png")]
        )
        _write_manifest(
            manifests_dir, "extract-0003.jsonl", [_make_record("/d/c.png", "c.png")]
        )

        result = assign_splits(str(manifests_dir), str(dest_dir))

        assert result.source_manifest_count == 3
        written = records_reader(result.split_manifest_path)
        assert {r.path for r in written} == {"/d/a.png", "/d/b.png", "/d/c.png"}

    def test_a_folder_with_no_manifest_raises_file_not_found_naming_the_folder(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits

        manifests_dir = tmp_path / "empty"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"

        with pytest.raises(FileNotFoundError, match=str(manifests_dir)):
            assign_splits(str(manifests_dir), str(dest_dir))

    def test_a_missing_folder_raises_file_not_found_naming_the_folder(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits

        manifests_dir = tmp_path / "does_not_exist"
        dest_dir = tmp_path / "out"

        with pytest.raises(FileNotFoundError, match=str(manifests_dir)):
            assign_splits(str(manifests_dir), str(dest_dir))

    def test_non_manifest_files_alongside_manifests_are_ignored(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir, "extract-0001.jsonl", [_make_record("/d/a.png", "a.png")]
        )
        (manifests_dir / "README.txt").write_text("not a manifest")
        (manifests_dir / "notes.md").write_text("also not a manifest")

        result = assign_splits(str(manifests_dir), str(dest_dir))

        written = records_reader(result.split_manifest_path)
        assert [r.path for r in written] == ["/d/a.png"]


class TestSplitStability:
    def test_a_path_present_in_both_runs_keeps_the_same_split_after_growth(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        first_dest = tmp_path / "out1"
        records = [
            _make_record(f"/d/img_{i:04d}.png", f"img_{i:04d}.png") for i in range(30)
        ]
        _write_manifest(manifests_dir, "extract-0001.jsonl", records)

        first = assign_splits(str(manifests_dir), str(first_dest))
        first_splits = {
            r.path: r.split for r in records_reader(first.split_manifest_path)
        }

        more_records = [
            _make_record(f"/d/img_new_{i:04d}.png", f"img_new_{i:04d}.png")
            for i in range(10)
        ]
        _write_manifest(manifests_dir, "extract-0002.jsonl", more_records)
        second_dest = tmp_path / "out2"

        second = assign_splits(str(manifests_dir), str(second_dest))
        second_splits = {
            r.path: r.split for r in records_reader(second.split_manifest_path)
        }

        for path, split in first_splits.items():
            assert second_splits[path] == split

    def test_default_ratios_match_the_previous_pipelines_split_for_a_filename(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_split, assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        record = _make_record("/d/known_patient.png", "known_patient.png")
        _write_manifest(manifests_dir, "extract-0001.jsonl", [record])

        result = assign_splits(str(manifests_dir), str(dest_dir))
        written = records_reader(result.split_manifest_path)

        expected = assign_split(
            "known_patient.png", [("train", 0.70), ("val", 0.15), ("test", 0.15)]
        )
        assert written[0].split == expected


class TestRatioValidation:
    def test_ratios_as_a_plain_mapping_raise_value_error(self, tmp_path: Path) -> None:
        from radiologist.etl import assign_splits

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir, "extract-0001.jsonl", [_make_record("/d/a.png", "a.png")]
        )

        with pytest.raises(ValueError, match="order"):
            assign_splits(
                str(manifests_dir),
                str(dest_dir),
                ratios={"train": 0.70, "val": 0.15, "test": 0.15},  # type: ignore[arg-type]
            )

    def test_ratios_not_summing_to_one_raise_value_error_with_observed_sum(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir, "extract-0001.jsonl", [_make_record("/d/a.png", "a.png")]
        )

        with pytest.raises(ValueError, match="0.8"):
            assign_splits(
                str(manifests_dir),
                str(dest_dir),
                ratios=[("train", 0.5), ("val", 0.3)],
            )


class TestDuplicateHandling:
    def test_the_same_source_path_in_two_manifests_is_deduplicated_to_one_record(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir,
            "extract-0001.jsonl",
            [_make_record("/d/dup.png", "dup.png", stats={"haralick_contrast": 1.0})],
        )
        _write_manifest(
            manifests_dir,
            "extract-0002.jsonl",
            [_make_record("/d/dup.png", "dup.png", stats={"haralick_contrast": 9.0})],
        )

        with caplog.at_level("WARNING"):
            result = assign_splits(str(manifests_dir), str(dest_dir))

        written = records_reader(result.split_manifest_path)
        assert len(written) == 1
        assert result.duplicate_count == 1
        assert any("1" in message for message in caplog.messages)

    def test_the_kept_duplicate_is_from_the_manifest_that_sorts_first_by_name(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir,
            "extract-0001.jsonl",
            [_make_record("/d/dup.png", "dup.png", stats={"haralick_contrast": 1.0})],
        )
        _write_manifest(
            manifests_dir,
            "extract-0002.jsonl",
            [_make_record("/d/dup.png", "dup.png", stats={"haralick_contrast": 9.0})],
        )

        result = assign_splits(str(manifests_dir), str(dest_dir))
        written = records_reader(result.split_manifest_path)

        assert written[0].stats["haralick_contrast"] == 1.0

    def test_distinct_paths_sharing_a_filename_are_both_kept_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir,
            "extract-0001.jsonl",
            [_make_record("/d/one/shared.png", "shared.png")],
        )
        _write_manifest(
            manifests_dir,
            "extract-0002.jsonl",
            [_make_record("/d/two/shared.png", "shared.png")],
        )

        with caplog.at_level("WARNING"):
            result = assign_splits(str(manifests_dir), str(dest_dir))

        written = records_reader(result.split_manifest_path)
        assert {r.path for r in written} == {"/d/one/shared.png", "/d/two/shared.png"}
        assert any("shared.png" in message for message in caplog.messages)


class TestExcludedRecords:
    def test_excluded_records_keep_their_flag_and_reason_and_still_get_a_split(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir,
            "extract-0001.jsonl",
            [
                _make_record(
                    "/d/bad.png",
                    "bad.png",
                    excluded=True,
                    exclusion_reason="iqr:haralick_contrast",
                )
            ],
        )

        result = assign_splits(str(manifests_dir), str(dest_dir))
        written = records_reader(result.split_manifest_path)

        assert written[0].excluded is True
        assert written[0].exclusion_reason == "iqr:haralick_contrast"
        assert written[0].split


class TestIdempotencyAndRunId:
    def test_rerun_over_unchanged_folder_writes_byte_identical_content(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir,
            "extract-0001.jsonl",
            [_make_record("/d/a.png", "a.png"), _make_record("/d/b.png", "b.png")],
        )

        first = assign_splits(str(manifests_dir), str(dest_dir))
        second = assign_splits(str(manifests_dir), str(dest_dir))

        assert first.split_manifest_path == second.split_manifest_path
        content1 = Path(first.split_manifest_path).read_bytes()
        content2 = Path(second.split_manifest_path).read_bytes()
        assert content1 == content2

    def test_adding_a_manifest_writes_a_different_file_leaving_the_first_in_place(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir, "extract-0001.jsonl", [_make_record("/d/a.png", "a.png")]
        )

        first = assign_splits(str(manifests_dir), str(dest_dir))
        _write_manifest(
            manifests_dir, "extract-0002.jsonl", [_make_record("/d/b.png", "b.png")]
        )
        second = assign_splits(str(manifests_dir), str(dest_dir))

        assert first.split_manifest_path != second.split_manifest_path
        assert Path(first.split_manifest_path).exists()

    def test_changing_ratios_writes_a_different_file_leaving_the_first_in_place(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir, "extract-0001.jsonl", [_make_record("/d/a.png", "a.png")]
        )

        first = assign_splits(
            str(manifests_dir),
            str(dest_dir),
            ratios=[("train", 0.70), ("val", 0.15), ("test", 0.15)],
        )
        second = assign_splits(
            str(manifests_dir),
            str(dest_dir),
            ratios=[("train", 0.60), ("val", 0.20), ("test", 0.20)],
        )

        assert first.split_manifest_path != second.split_manifest_path
        assert Path(first.split_manifest_path).exists()


class TestReportingAndReadability:
    def test_counts_by_split_match_the_records_actually_written(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        records = [
            _make_record(f"/d/img_{i:04d}.png", f"img_{i:04d}.png") for i in range(40)
        ]
        _write_manifest(manifests_dir, "extract-0001.jsonl", records)

        result = assign_splits(str(manifests_dir), str(dest_dir))
        written = records_reader(result.split_manifest_path)

        actual_counts: dict = {}
        for r in written:
            actual_counts[r.split] = actual_counts.get(r.split, 0) + 1
        assert result.counts_by_split == actual_counts
        assert sum(result.counts_by_split.values()) == result.record_count
        assert result.record_count == len(written)

    def test_storage_options_reach_every_filesystem_access(
        self, tmp_path: Path
    ) -> None:
        from radiologist.etl import assign_splits, records_reader

        manifests_dir = tmp_path / "manifests"
        manifests_dir.mkdir()
        dest_dir = tmp_path / "out"
        _write_manifest(
            manifests_dir, "extract-0001.jsonl", [_make_record("/d/a.png", "a.png")]
        )

        # LocalFileSystem accepts (and ignores) an "auto_mkdir" kwarg; a bogus
        # unsupported kwarg would raise if it were not actually forwarded.
        result = assign_splits(
            str(manifests_dir),
            str(dest_dir),
            storage_options={"auto_mkdir": True},
        )

        written = records_reader(
            result.split_manifest_path, storage_options={"auto_mkdir": True}
        )
        assert len(written) == 1
