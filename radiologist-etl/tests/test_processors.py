# MIT License
#
# Copyright (c) 2026 @CedrickArmel, @TaxelleT, @Yeyecodes
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

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from radiologist.etl.processors import StatsProcessor, _resolve_mask
from radiologist.etl.stats import make_haralick
from radiologist.utils.readers import ImageReader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_png(path: Path) -> None:
    """Save a 10x10 random RGB PNG to path."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_white_mask_png(path: Path) -> None:
    """Save a 10x10 all-white mask PNG; border pixels are nonzero."""
    arr = np.full((10, 10, 3), 255, dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_interior_mask_png(path: Path) -> None:
    """Save a 10x10 mask with nonzero only in the interior."""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[3:7, 3:7] = 255
    Image.fromarray(arr).save(path)


def _make_border_mask_png(path: Path) -> None:
    """Save a 10x10 mask with nonzero on the first row."""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[0, :] = 255
    Image.fromarray(arr).save(path)


def _build_two_class_tree(base: Path) -> Path:
    """Create images/NORMAL/ and images/ABNORMAL/ each with 1 image."""
    images_root = base / "images"
    for cls in ("NORMAL", "ABNORMAL"):
        (images_root / cls).mkdir(parents=True)
        _make_rgb_png(images_root / cls / "img001.png")
    return images_root


# ---------------------------------------------------------------------------
# AC1: run returns one record per image
# ---------------------------------------------------------------------------


def test_run_returns_one_record_per_image(tmp_path: Path) -> None:
    images_root = _build_two_class_tree(tmp_path)
    reader = ImageReader(str(images_root))
    processor = StatsProcessor(extractors=[make_haralick(features=["contrast"])])
    records = processor.run(reader, manifest_id="test-id-0001")
    assert len(records) == 2


# ---------------------------------------------------------------------------
# AC2: each record has the correct manifest_id
# ---------------------------------------------------------------------------


def test_run_records_have_correct_manifest_id(tmp_path: Path) -> None:
    images_root = _build_two_class_tree(tmp_path)
    reader = ImageReader(str(images_root))
    processor = StatsProcessor(extractors=[make_haralick(features=["contrast"])])
    records = processor.run(reader, manifest_id="run-xyz-9999")
    assert all(r.manifest_id == "run-xyz-9999" for r in records)


# ---------------------------------------------------------------------------
# AC3: each record's label matches parent directory name
# ---------------------------------------------------------------------------


def test_run_record_label_matches_parent_directory_name(tmp_path: Path) -> None:
    images_root = _build_two_class_tree(tmp_path)
    reader = ImageReader(str(images_root))
    processor = StatsProcessor(extractors=[make_haralick(features=["contrast"])])
    records = processor.run(reader, manifest_id="test-id-0001")
    labels = {r.label for r in records}
    assert labels == {"NORMAL", "ABNORMAL"}


# ---------------------------------------------------------------------------
# AC4: each record's stats keys come from the configured extractors
# ---------------------------------------------------------------------------


def test_run_record_stats_keys_come_from_extractors(tmp_path: Path) -> None:
    images_root = _build_two_class_tree(tmp_path)
    reader = ImageReader(str(images_root))
    processor = StatsProcessor(
        extractors=[make_haralick(features=["contrast", "energy"])]
    )
    records = processor.run(reader, manifest_id="test-id-0001")
    for record in records:
        assert set(record.stats.keys()) == {"haralick_contrast", "haralick_energy"}


# ---------------------------------------------------------------------------
# AC5: when masks_root provided and mask file exists, lung_out_of_frame not None
# ---------------------------------------------------------------------------


def test_run_lung_out_of_frame_not_none_when_mask_exists(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    masks_root = tmp_path / "masks"
    (images_root / "NORMAL").mkdir(parents=True)
    (masks_root / "NORMAL").mkdir(parents=True)
    _make_rgb_png(images_root / "NORMAL" / "img001.png")
    _make_white_mask_png(masks_root / "NORMAL" / "img001.png")

    reader = ImageReader(str(images_root))
    processor = StatsProcessor(extractors=[make_haralick(features=["contrast"])])
    records = processor.run(
        reader, manifest_id="test-id-0001", masks_root=str(masks_root)
    )
    assert all(r.lung_out_of_frame is not None for r in records)


# ---------------------------------------------------------------------------
# AC6: when masks_root=None, all records have lung_out_of_frame=None
# ---------------------------------------------------------------------------


def test_run_lung_out_of_frame_is_none_when_no_masks_root(tmp_path: Path) -> None:
    images_root = _build_two_class_tree(tmp_path)
    reader = ImageReader(str(images_root))
    processor = StatsProcessor(extractors=[make_haralick(features=["contrast"])])
    records = processor.run(reader, manifest_id="test-id-0001", masks_root=None)
    assert all(r.lung_out_of_frame is None for r in records)


# ---------------------------------------------------------------------------
# AC7: border-touching mask -> lung_out_of_frame=True
# ---------------------------------------------------------------------------


def test_run_lung_out_of_frame_true_for_border_touching_mask(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    masks_root = tmp_path / "masks"
    (images_root / "NORMAL").mkdir(parents=True)
    (masks_root / "NORMAL").mkdir(parents=True)
    _make_rgb_png(images_root / "NORMAL" / "img001.png")
    _make_border_mask_png(masks_root / "NORMAL" / "img001.png")

    reader = ImageReader(str(images_root))
    processor = StatsProcessor(extractors=[make_haralick(features=["contrast"])])
    records = processor.run(
        reader, manifest_id="test-id-0001", masks_root=str(masks_root)
    )
    assert len(records) == 1
    assert records[0].lung_out_of_frame is True


# ---------------------------------------------------------------------------
# AC8: interior-only mask -> lung_out_of_frame=False
# ---------------------------------------------------------------------------


def test_run_lung_out_of_frame_false_for_interior_only_mask(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    masks_root = tmp_path / "masks"
    (images_root / "NORMAL").mkdir(parents=True)
    (masks_root / "NORMAL").mkdir(parents=True)
    _make_rgb_png(images_root / "NORMAL" / "img001.png")
    _make_interior_mask_png(masks_root / "NORMAL" / "img001.png")

    reader = ImageReader(str(images_root))
    processor = StatsProcessor(extractors=[make_haralick(features=["contrast"])])
    records = processor.run(
        reader, manifest_id="test-id-0001", masks_root=str(masks_root)
    )
    assert len(records) == 1
    assert records[0].lung_out_of_frame is False


# ---------------------------------------------------------------------------
# AC9: per-image errors are caught and logged -- run does not abort
# ---------------------------------------------------------------------------


def test_run_skips_failed_images_and_returns_remaining_records(
    tmp_path: Path,
) -> None:
    images_root = tmp_path / "images"
    (images_root / "NORMAL").mkdir(parents=True)
    _make_rgb_png(images_root / "NORMAL" / "img001.png")
    corrupt = images_root / "NORMAL" / "img002.png"
    corrupt.write_bytes(b"this is not a png")

    reader = ImageReader(str(images_root))
    processor = StatsProcessor(extractors=[make_haralick(features=["contrast"])])
    records = processor.run(reader, manifest_id="test-id-0001")
    assert len(records) == 1


# ---------------------------------------------------------------------------
# AC10: Integration -- two class dirs, 2 images each, masks provided
# ---------------------------------------------------------------------------


def test_integration_two_classes_two_images_each_with_masks(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    masks_root = tmp_path / "masks"
    for cls in ("NORMAL", "ABNORMAL"):
        (images_root / cls).mkdir(parents=True)
        (masks_root / cls).mkdir(parents=True)
        for idx in range(1, 3):
            fname = f"img{idx:03d}.png"
            _make_rgb_png(images_root / cls / fname)
            _make_white_mask_png(masks_root / cls / fname)

    reader = ImageReader(str(images_root))
    processor = StatsProcessor(
        extractors=[make_haralick(features=["contrast", "energy"])]
    )
    records = processor.run(
        reader, manifest_id="integration-run-001", masks_root=str(masks_root)
    )

    assert len(records) == 4

    for record in records:
        assert record.manifest_id == "integration-run-001"
        assert record.label in {"NORMAL", "ABNORMAL"}
        assert "haralick_contrast" in record.stats
        assert "haralick_energy" in record.stats


# ---------------------------------------------------------------------------
# FIX 6: _resolve_mask boundary check — prefix-only match must return None
# ---------------------------------------------------------------------------


def test_resolve_mask_returns_none_when_images_root_is_prefix_of_different_dir(
    tmp_path: Path,
) -> None:
    """/data_backup/NORMAL/img.png with images_root=/data must return None.

    Creates the would-be-wrong mask file to confirm the function rejects via
    boundary check, not just FileNotFoundError.
    """
    masks_root = tmp_path / "masks"
    # Create the wrong-path mask file that the old code would compute
    wrong_mask_dir = masks_root / "_backup" / "NORMAL"
    wrong_mask_dir.mkdir(parents=True)
    wrong_mask_arr = np.zeros((10, 10, 3), dtype=np.uint8)
    from PIL import Image as _PIL_Image

    _PIL_Image.fromarray(wrong_mask_arr).save(str(wrong_mask_dir / "img.png"))

    result = _resolve_mask(
        image_path="/data_backup/NORMAL/img.png",
        images_root="/data",
        masks_root=str(masks_root),
        storage_options=None,
    )
    assert result is None
