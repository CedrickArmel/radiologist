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

"""Behavioral tests for the extract stage (#183): explicit file listing ->
stats + quality filters -> one batch manifest per run."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# extract()'s default mapper is a real process pool (spawn context). Custom
# StatExtractor callables defined in this test module must be importable by
# dotted path from the freshly-started child interpreter, which only
# inherits PYTHONPATH, not this process's in-memory sys.path mutations.
# See feedback_spawn_pool_test_module_pythonpath memory note.
_TESTS_PARENT = str(Path(__file__).resolve().parents[1])
if _TESTS_PARENT not in sys.path:
    sys.path.insert(0, _TESTS_PARENT)
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
_pythonpath_parts = [_TESTS_PARENT] + (
    _existing_pythonpath.split(os.pathsep) if _existing_pythonpath else []
)
os.environ["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(_pythonpath_parts))


def _controlled_outlier_stat(image, metadata, mask=None):
    """Module-level (picklable) StatExtractor: value keyed off the filename
    only, so it needs no captured state — safe to cross a process boundary.

    Checks ``filename`` (not the full ``path``) for the "outlier" marker:
    pytest's ``tmp_path`` directory is itself named after the test function,
    so a substring match against the full path can false-positive when the
    test name contains "outlier" too.
    """
    value = 1000.0 if metadata["filename"].startswith("outlier") else 1.0
    return {"outlier_col": value}


def _write_listing(tmp_path: Path, paths, name: str = "listing.txt") -> str:
    listing_path = tmp_path / name
    listing_path.write_text("\n".join(paths) + "\n")
    return str(listing_path)


def _read_manifest_lines(path: str) -> list[dict]:
    with open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _all_image_paths(image_dir: Path) -> list[str]:
    return [str(p) for p in sorted(image_dir.rglob("*.png"))]


# --- read_file_list -------------------------------------------------------


def test_read_file_list_returns_uris_in_file_order(tmp_path):
    from radiologist.etl.extract import read_file_list

    listing = _write_listing(tmp_path, ["a.png", "b.png", "c.png"])

    assert read_file_list(listing) == ["a.png", "b.png", "c.png"]


def test_read_file_list_skips_blank_and_comment_lines_and_strips_whitespace(tmp_path):
    from radiologist.etl.extract import read_file_list

    listing_path = tmp_path / "listing.txt"
    listing_path.write_text("  a.png  \n\n# a comment\nb.png\n   \n#another\nc.png\n")

    assert read_file_list(str(listing_path)) == ["a.png", "b.png", "c.png"]


def test_read_file_list_raises_value_error_when_no_entries(tmp_path):
    from radiologist.etl.extract import read_file_list

    listing_path = tmp_path / "empty.txt"
    listing_path.write_text("\n\n# only comments\n\n")

    with pytest.raises(ValueError):
        read_file_list(str(listing_path))


def test_read_file_list_raises_file_not_found_for_missing_listing(tmp_path):
    from radiologist.etl.extract import read_file_list

    missing = str(tmp_path / "does-not-exist.txt")

    with pytest.raises(FileNotFoundError, match="does-not-exist.txt"):
        read_file_list(missing)


# --- extract(): core manifest contents -------------------------------------


def test_extract_writes_one_manifest_with_one_record_per_listed_image_in_order(
    image_dir, tmp_path
):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result = extract(listing, destination, images_root=str(image_dir))

    rows = _read_manifest_lines(result.manifest_path)
    assert [r["path"] for r in rows] == paths


def test_extract_ignores_images_not_named_in_the_listing(image_dir, tmp_path):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    named = paths[:2]
    listing = _write_listing(tmp_path, named)
    destination = str(tmp_path / "dest")

    result = extract(listing, destination, images_root=str(image_dir))

    rows = _read_manifest_lines(result.manifest_path)
    assert {r["path"] for r in rows} == set(named)


def test_every_record_carries_empty_split_path_filename_label_and_stats(
    image_dir, tmp_path
):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result = extract(listing, destination, images_root=str(image_dir))

    rows = _read_manifest_lines(result.manifest_path)
    for row in rows:
        assert row["split"] == ""
        assert row["path"] in paths
        assert row["filename"] == Path(row["path"]).name
        assert row["label"] == Path(row["path"]).parent.name
        assert "haralick_mean" in row


def test_records_carry_lung_asymmetry_stats_when_masks_available(
    image_dir, mask_dir, tmp_path
):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result = extract(
        listing,
        destination,
        images_root=str(image_dir),
        masks_root=str(mask_dir),
    )

    rows = _read_manifest_lines(result.manifest_path)
    assert all("asymmetry_ratio" in r for r in rows)


def test_no_masks_means_no_record_flagged_out_of_frame_and_stage_completes(
    image_dir, tmp_path
):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result = extract(listing, destination, images_root=str(image_dir))

    rows = _read_manifest_lines(result.manifest_path)
    assert all(r["lung_out_of_frame"] is None for r in rows)
    assert result.total == len(paths)


# --- quality filters --------------------------------------------------------


def test_iqr_outlier_is_flagged_excluded_and_still_present(image_dir, tmp_path):
    import shutil

    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    outlier_src = Path(paths[0])
    outlier_path = str(outlier_src.with_name("outlier.png"))
    shutil.copy(outlier_src, outlier_path)
    paths = paths[1:] + [outlier_path]

    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result = extract(
        listing,
        destination,
        images_root=str(image_dir),
        extractors=[_controlled_outlier_stat],
        iqr_columns=["outlier_col"],
    )

    rows = _read_manifest_lines(result.manifest_path)
    assert len(rows) == len(paths)
    flagged = [r for r in rows if r["path"] == outlier_path]
    assert flagged[0]["excluded"] is True
    assert "iqr:outlier_col" in flagged[0]["exclusion_reason"]
    assert result.excluded == 1


def test_lung_out_of_frame_and_iqr_reasons_combine_when_both_apply(
    image_dir, mask_dir, tmp_path
):
    import shutil

    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)

    # NORMAL masks touch the border (out of frame, see mask_dir fixture).
    # Duplicate one NORMAL image + its mirrored mask under a filename
    # containing "outlier" so _controlled_outlier_stat also flags it.
    normal_src = next(p for p in paths if Path(p).parent.name == "NORMAL")
    outlier_path = str(Path(normal_src).with_name("outlier.png"))
    shutil.copy(normal_src, outlier_path)
    mask_src = mask_dir / "NORMAL" / Path(normal_src).name
    shutil.copy(mask_src, mask_dir / "NORMAL" / "outlier.png")

    paths = [p for p in paths if p != normal_src] + [outlier_path]
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result = extract(
        listing,
        destination,
        images_root=str(image_dir),
        masks_root=str(mask_dir),
        extractors=[_controlled_outlier_stat],
        iqr_columns=["outlier_col"],
    )

    rows = _read_manifest_lines(result.manifest_path)
    outlier_rows = [r for r in rows if r["path"] == outlier_path]
    assert outlier_rows[0]["excluded"] is True
    assert "lung_out_of_frame" in outlier_rows[0]["exclusion_reason"]
    assert "iqr:outlier_col" in outlier_rows[0]["exclusion_reason"]

    abnormal_path = next(p for p in paths if Path(p).parent.name == "ABNORMAL")
    abnormal_rows = [r for r in rows if r["path"] == abnormal_path]
    assert abnormal_rows[0]["excluded"] is False


# --- masks_root/images_root invariant --------------------------------------


def test_masks_root_without_images_root_raises_value_error(image_dir, tmp_path):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    with pytest.raises(ValueError, match="images_root"):
        extract(listing, destination, masks_root="/some/masks")


# --- run-id / manifest naming stability -------------------------------------


def test_same_listing_and_settings_write_to_the_same_manifest_name(image_dir, tmp_path):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result1 = extract(listing, destination, images_root=str(image_dir))
    result2 = extract(listing, destination, images_root=str(image_dir))

    assert result1.manifest_path == result2.manifest_path
    assert result1.run_id == result2.run_id


def test_changing_a_filter_setting_writes_a_different_manifest_leaving_first_in_place(
    image_dir, tmp_path
):
    from radiologist.etl.extract import extract
    from radiologist.etl.manifest import records_reader

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result1 = extract(listing, destination, images_root=str(image_dir), iqr_factor=1.5)
    result2 = extract(listing, destination, images_root=str(image_dir), iqr_factor=3.0)

    assert result1.manifest_path != result2.manifest_path
    # first manifest is untouched by the second run
    assert len(records_reader(result1.manifest_path)) == len(paths)


def test_different_worker_counts_or_batch_sizes_write_identical_manifest_bytes(
    image_dir, tmp_path
):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result1 = extract(
        listing, destination, images_root=str(image_dir), workers=1, batch_size=1
    )
    result2 = extract(
        listing, destination, images_root=str(image_dir), workers=2, batch_size=64
    )

    assert result1.manifest_path == result2.manifest_path
    with open(result1.manifest_path, "rb") as f:
        content1 = f.read()
    with open(result2.manifest_path, "rb") as f:
        content2 = f.read()
    assert content1 == content2


def test_two_different_listings_leave_two_manifests_side_by_side(image_dir, tmp_path):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    destination = str(tmp_path / "dest")

    listing1 = _write_listing(tmp_path, paths[:2], name="listing1.txt")
    listing2 = _write_listing(tmp_path, paths[2:], name="listing2.txt")

    result1 = extract(listing1, destination, images_root=str(image_dir))
    result2 = extract(listing2, destination, images_root=str(image_dir))

    assert result1.manifest_path != result2.manifest_path
    assert Path(result1.manifest_path).exists()
    assert Path(result2.manifest_path).exists()


# --- failure handling --------------------------------------------------------


def test_unreadable_image_with_zero_tolerance_raises_and_writes_no_manifest(
    image_dir, tmp_path
):
    from radiologist.etl.extract import ExtractionFailureError, extract

    paths = _all_image_paths(image_dir)
    bad_path = str(tmp_path / "missing.png")
    listing = _write_listing(tmp_path, paths + [bad_path])
    destination = str(tmp_path / "dest")

    with pytest.raises(ExtractionFailureError, match="missing.png"):
        extract(
            listing,
            destination,
            images_root=str(image_dir),
            max_failure_rate=0.0,
        )

    assert not Path(destination).exists() or not any(Path(destination).iterdir())


def test_partial_failure_within_tolerance_completes_and_reports_counts(
    image_dir, tmp_path
):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)[:3]
    bad_path = str(tmp_path / "missing.png")
    listing = _write_listing(tmp_path, paths + [bad_path])
    destination = str(tmp_path / "dest")

    result = extract(
        listing,
        destination,
        images_root=str(image_dir),
        max_failure_rate=0.5,
    )

    assert result.total == 4
    assert result.succeeded == 3
    assert result.failed == 1
    assert result.failure_rate == pytest.approx(0.25)

    rows = _read_manifest_lines(result.manifest_path)
    assert len(rows) == 3
    assert {r["path"] for r in rows} == set(paths)


# --- empty / missing listing -------------------------------------------------


def test_listing_resolving_to_no_entries_raises_value_error(tmp_path):
    from radiologist.etl.extract import extract

    listing_path = tmp_path / "empty.txt"
    listing_path.write_text("\n# only comments\n")
    destination = str(tmp_path / "dest")

    with pytest.raises(ValueError):
        extract(str(listing_path), destination)


def test_missing_listing_uri_raises_file_not_found(tmp_path):
    from radiologist.etl.extract import extract

    missing = str(tmp_path / "does-not-exist.txt")
    destination = str(tmp_path / "dest")

    with pytest.raises(FileNotFoundError, match="does-not-exist.txt"):
        extract(missing, destination)


# --- batching ------------------------------------------------------------


def test_more_images_than_one_batch_still_yields_one_record_per_image(
    image_dir, tmp_path
):
    from radiologist.etl.extract import extract

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result = extract(listing, destination, images_root=str(image_dir), batch_size=1)

    rows = _read_manifest_lines(result.manifest_path)
    assert len(rows) == len(paths)
    assert result.total == len(paths)
    assert result.succeeded == len(paths)


# --- ExtractResult / storage_options ----------------------------------------


def test_result_counts_match_the_written_manifest(image_dir, mask_dir, tmp_path):
    from radiologist.etl.extract import extract
    from radiologist.etl.manifest import records_reader

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")

    result = extract(
        listing,
        destination,
        images_root=str(image_dir),
        masks_root=str(mask_dir),
    )

    written = records_reader(result.manifest_path)
    assert result.total == len(paths)
    assert result.succeeded == len(written)
    assert result.excluded == sum(1 for r in written if r.excluded)
    assert Path(result.manifest_path).exists()


def test_storage_options_reach_listing_image_mask_and_destination_access(
    image_dir, mask_dir, tmp_path, monkeypatch
):
    import fsspec

    from radiologist.etl.extract import extract

    seen: list[tuple[str, dict]] = []
    real_url_to_fs = fsspec.url_to_fs

    def spy_url_to_fs(uri, **kwargs):
        seen.append((str(uri), dict(kwargs)))
        return real_url_to_fs(uri, **kwargs)

    monkeypatch.setattr(fsspec, "url_to_fs", spy_url_to_fs)

    paths = _all_image_paths(image_dir)
    listing = _write_listing(tmp_path, paths)
    destination = str(tmp_path / "dest")
    storage_options = {"auto_mkdir": True}

    extract(
        listing,
        destination,
        images_root=str(image_dir),
        masks_root=str(mask_dir),
        storage_options=storage_options,
    )

    # every call touching a path under our test tree must have carried the
    # storage_options through
    relevant = [
        (uri, kwargs)
        for uri, kwargs in seen
        if str(tmp_path) in uri or uri.startswith(destination)
    ]
    assert relevant, "expected at least one fsspec.url_to_fs call to be observed"
    assert all(kwargs == storage_options for _, kwargs in relevant)
