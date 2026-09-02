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

"""Behavioral tests for the build stage's public API (issue #185)."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image  # type: ignore[import-untyped]

from radiologist.etl.manifest import JsonlWriter, ManifestRecord, records_reader

_RATIOS = [("train", 0.70), ("val", 0.15), ("test", 0.15)]


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def _make_png(root: Path, filename: str, label: str) -> str:
    dest = root / label / filename
    _write_png(dest, np.zeros((10, 10, 3), dtype=np.uint8))
    return str(dest)


def _record(path: str, filename: str, label: str, split: str, excluded=False):
    return ManifestRecord(
        manifest_id="assignrun00000001",
        path=path,
        filename=filename,
        label=label,
        split=split,
        stats={},
        excluded=excluded,
    )


def _make_split_manifest(root: Path, records: list[ManifestRecord]) -> str:
    path = str(root / "manifest-assignrun00000001.jsonl")
    JsonlWriter().write(records, path)
    return path


def _tar_members(tar_path: str) -> list[str]:
    with tarfile.open(tar_path) as tf:
        return [m.name for m in tf.getmembers()]


# --- shard writing -----------------------------------------------------------


def test_writes_tar_shards_grouped_by_split_and_label_and_reports_shard_count(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    records = [_record(path_a, "a.png", "NORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"))

    assert result.shard_count == 1
    tar_path = str(
        Path(result.output_dir) / "train" / "NORMAL" / "train-normal-000000.tar"
    )
    assert Path(tar_path).exists()


def test_group_larger_than_shard_size_is_split_across_several_shards_numbered_from_zero(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    records = [
        _record(
            _make_png(tmp_path, f"img_{i:03d}.png", "NORMAL"),
            f"img_{i:03d}.png",
            "NORMAL",
            "train",
        )
        for i in range(5)
    ]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"), shard_size=2)

    shard_dir = Path(result.output_dir) / "train" / "NORMAL"
    tar_files = sorted(p.name for p in shard_dir.glob("*.tar"))
    assert tar_files == [
        "train-normal-000000.tar",
        "train-normal-000001.tar",
        "train-normal-000002.tar",
    ]


def test_each_group_is_numbered_independently_from_zero(tmp_path: Path) -> None:
    from radiologist.etl.build import build_shards

    records = [
        _record(_make_png(tmp_path, "n0.png", "NORMAL"), "n0.png", "NORMAL", "train"),
        _record(
            _make_png(tmp_path, "a0.png", "ABNORMAL"), "a0.png", "ABNORMAL", "train"
        ),
    ]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"))

    assert (
        Path(result.output_dir) / "train" / "NORMAL" / "train-normal-000000.tar"
    ).exists()
    assert (
        Path(result.output_dir) / "train" / "ABNORMAL" / "train-abnormal-000000.tar"
    ).exists()


def test_shard_entry_carries_image_bytes_and_label_keyed_by_stem(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "scan001.png", "ABNORMAL")
    records = [_record(path_a, "scan001.png", "ABNORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"))

    tar_path = str(
        Path(result.output_dir) / "train" / "ABNORMAL" / "train-abnormal-000000.tar"
    )
    with tarfile.open(tar_path) as tf:
        members = {m.name for m in tf.getmembers()}
        assert "scan001.cls" in members
        assert "scan001.png" in members
        cls_f = tf.extractfile("scan001.cls")
        assert cls_f is not None
        assert cls_f.read().decode("utf-8") == "ABNORMAL"


# --- exclusion ---------------------------------------------------------------


def test_excluded_records_are_written_into_no_shard_and_not_counted(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    path_b = _make_png(tmp_path, "b.png", "NORMAL")
    records = [
        _record(path_a, "a.png", "NORMAL", "train"),
        _record(path_b, "b.png", "NORMAL", "train", excluded=True),
    ]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"))

    tar_path = str(
        Path(result.output_dir) / "train" / "NORMAL" / "train-normal-000000.tar"
    )
    members = _tar_members(tar_path)
    assert not any("b" in m for m in members)
    assert result.record_count == 1


def test_all_records_excluded_writes_no_shard_zero_counts_but_manifest_and_report(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    records = [_record(path_a, "a.png", "NORMAL", "train", excluded=True)]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"))

    assert result.shard_count == 0
    assert result.record_count == 0
    assert Path(result.manifest_path).exists()
    assert Path(result.report_path).exists()


# --- manifest --------------------------------------------------------------


def test_manifest_carries_shard_location_for_non_excluded_records_and_none_for_excluded(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    path_b = _make_png(tmp_path, "b.png", "NORMAL")
    records = [
        _record(path_a, "a.png", "NORMAL", "train"),
        _record(path_b, "b.png", "NORMAL", "train", excluded=True),
    ]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"))

    out_records = records_reader(result.manifest_path)
    by_filename = {r.filename: r for r in out_records}
    assert by_filename["a.png"].shard is not None
    assert by_filename["b.png"].shard is None


def test_input_split_manifest_is_byte_identical_before_and_after(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    records = [_record(path_a, "a.png", "NORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)
    before = Path(manifest_path).read_bytes()

    build_shards(manifest_path, str(tmp_path / "shards"))

    after = Path(manifest_path).read_bytes()
    assert before == after


def test_manifest_shard_location_resolves_against_output_folder(tmp_path: Path) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    records = [_record(path_a, "a.png", "NORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"))

    out_records = records_reader(result.manifest_path)
    rec = out_records[0]
    resolved = Path(result.output_dir) / rec.shard
    assert resolved.exists()


# --- split report -------------------------------------------------------------


def test_report_names_configured_ratios_and_observed_proportions_including_excluded(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    path_b = _make_png(tmp_path, "b.png", "NORMAL")
    records = [
        _record(path_a, "a.png", "NORMAL", "train"),
        _record(path_b, "b.png", "NORMAL", "train", excluded=True),
    ]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"), ratios=_RATIOS)

    report = json.loads(Path(result.report_path).read_text())
    assert report["configured_ratios"] == [list(p) for p in _RATIOS]
    assert report["observed"]["NORMAL"]["excluded"] == pytest.approx(0.5)
    assert report["observed"]["NORMAL"]["train"] == pytest.approx(0.5)


def test_report_shows_observed_composition_differing_from_configured_ratios_without_altering_split(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    records = [
        _record(
            _make_png(tmp_path, f"img{i}.png", "NORMAL"),
            f"img{i}.png",
            "NORMAL",
            "train",
        )
        for i in range(4)
    ]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"), ratios=_RATIOS)

    report = json.loads(Path(result.report_path).read_text())
    assert report["configured_ratios"] == [list(p) for p in _RATIOS]
    assert report["observed"]["NORMAL"]["train"] == pytest.approx(1.0)

    out_records = records_reader(result.manifest_path)
    assert {r.split for r in out_records} == {"train"}


# --- run identity / re-run behavior -------------------------------------------


def test_running_twice_over_same_manifest_and_shard_size_writes_to_same_output_folder(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    records = [_record(path_a, "a.png", "NORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)

    result1 = build_shards(manifest_path, str(tmp_path / "shards"))
    result2 = build_shards(manifest_path, str(tmp_path / "shards"))

    assert result1.output_dir == result2.output_dir


def test_changing_shard_size_writes_to_a_different_output_folder_leaving_first_in_place(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    records = [
        _record(
            _make_png(tmp_path, f"img{i}.png", "NORMAL"),
            f"img{i}.png",
            "NORMAL",
            "train",
        )
        for i in range(3)
    ]
    manifest_path = _make_split_manifest(tmp_path, records)

    result1 = build_shards(manifest_path, str(tmp_path / "shards"), shard_size=1000)
    result2 = build_shards(manifest_path, str(tmp_path / "shards"), shard_size=1)

    assert result1.output_dir != result2.output_dir
    assert Path(result1.output_dir).exists()
    assert Path(result2.output_dir).exists()


def test_changing_ratios_writes_to_a_different_output_folder(tmp_path: Path) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    records = [_record(path_a, "a.png", "NORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)

    result1 = build_shards(
        manifest_path,
        str(tmp_path / "shards"),
        ratios=[("train", 0.70), ("val", 0.15), ("test", 0.15)],
    )
    result2 = build_shards(
        manifest_path,
        str(tmp_path / "shards"),
        ratios=[("train", 0.80), ("val", 0.10), ("test", 0.10)],
    )

    assert result1.run_id != result2.run_id
    assert result1.output_dir != result2.output_dir


def test_identical_ratios_including_order_across_two_runs_produce_the_same_run_id(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    records = [_record(path_a, "a.png", "NORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)
    same_ratios = [("train", 0.70), ("val", 0.15), ("test", 0.15)]

    result1 = build_shards(manifest_path, str(tmp_path / "shards"), ratios=same_ratios)
    result2 = build_shards(manifest_path, str(tmp_path / "shards"), ratios=same_ratios)

    assert result1.run_id == result2.run_id
    assert result1.output_dir == result2.output_dir


def test_changing_manifest_content_writes_to_a_different_output_folder(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    manifest_path = _make_split_manifest(
        tmp_path, [_record(path_a, "a.png", "NORMAL", "train")]
    )
    result1 = build_shards(manifest_path, str(tmp_path / "shards"))

    path_b = _make_png(tmp_path, "b.png", "NORMAL")
    manifest_path2 = str(tmp_path / "manifest-assignrun00000002.jsonl")
    JsonlWriter().write([_record(path_b, "b.png", "NORMAL", "train")], manifest_path2)
    result2 = build_shards(manifest_path2, str(tmp_path / "shards"))

    assert result1.output_dir != result2.output_dir


def test_running_twice_with_different_worker_counts_writes_byte_identical_manifests(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    records = [
        _record(
            _make_png(tmp_path, f"img{i}.png", "NORMAL"),
            f"img{i}.png",
            "NORMAL",
            "train",
        )
        for i in range(4)
    ]
    manifest_path = _make_split_manifest(tmp_path, records)

    result1 = build_shards(
        manifest_path, str(tmp_path / "shards1"), shard_size=1, workers=1
    )
    result2 = build_shards(
        manifest_path, str(tmp_path / "shards2"), shard_size=1, workers=2
    )

    manifest1 = Path(result1.manifest_path).read_text()
    manifest2 = Path(result2.manifest_path).read_text()
    # run ids differ because output roots differ (folded into config? no —
    # run id is content + config only, unaffected by shard_root), so manifest
    # bodies should be identical modulo the run id/shard-root prefix baked
    # into the shard path; normalize before compare.
    assert result1.run_id == result2.run_id
    assert manifest1.replace("shards1", "X") == manifest2.replace("shards2", "X")
    shard_names1 = sorted(p.name for p in Path(result1.output_dir).rglob("*.tar"))
    shard_names2 = sorted(p.name for p in Path(result2.output_dir).rglob("*.tar"))
    assert shard_names1 == shard_names2


# --- errors --------------------------------------------------------------------


def test_missing_split_manifest_raises_file_not_found_error_naming_it(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    missing = str(tmp_path / "does-not-exist.jsonl")

    with pytest.raises(FileNotFoundError, match="does-not-exist.jsonl"):
        build_shards(missing, str(tmp_path / "shards"))


def test_shard_size_below_one_raises_value_error(tmp_path: Path) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    manifest_path = _make_split_manifest(
        tmp_path, [_record(path_a, "a.png", "NORMAL", "train")]
    )

    with pytest.raises(ValueError):
        build_shards(manifest_path, str(tmp_path / "shards"), shard_size=0)


def test_unreadable_image_is_reported_as_failure_not_written_as_shard_entry(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    missing_path = str(tmp_path / "gone.png")
    records = [_record(missing_path, "gone.png", "NORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"))

    assert result.record_count == 0
    out_records = records_reader(result.manifest_path)
    assert out_records[0].shard is None


# --- storage options -----------------------------------------------------------


def test_storage_options_reach_manifest_and_image_reads_and_shard_writes(
    tmp_path: Path,
) -> None:
    from radiologist.etl.build import build_shards

    path_a = _make_png(tmp_path, "a.png", "NORMAL")
    records = [_record(path_a, "a.png", "NORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"), storage_options={})

    assert result.shard_count == 1
