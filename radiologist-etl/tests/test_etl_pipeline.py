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

import inspect
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from radiologist.etl.pipeline import _build_shards, compute_run_id, etl_flow, main


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def _build_image_tree(root: Path, n_per_class: int = 2) -> Path:
    """Create images/NORMAL/ and images/ABNORMAL/ each with n_per_class PNGs."""
    images = root / "images"
    rng = np.random.default_rng(7)
    for label in ("NORMAL", "ABNORMAL"):
        for i in range(n_per_class):
            _write_png(
                images / label / f"img{i:03d}.png",
                rng.integers(0, 256, (10, 10, 3), dtype=np.uint8),
            )
    return images


def _build_border_mask_tree(root: Path, images_root: Path) -> Path:
    """Create masks/ tree mirroring images_root; all NORMAL masks touch the border."""
    masks = root / "masks"
    border = np.zeros((10, 10, 3), dtype=np.uint8)
    border[0, :] = 255
    for label in ("NORMAL", "ABNORMAL"):
        src_dir = images_root / label
        for img_path in sorted(src_dir.glob("*.png")):
            _write_png(masks / label / img_path.name, border)
    return masks


def _minimal_cfg(
    images_root: Path,
    destination: Path,
    artifact_dir: Path,
    *,
    masks_root: Path | None = None,
    run_label: str = "test-run",
    resume_from_parquet: str | None = None,
    resume_from_filtered: str | None = None,
    resume_from_split: str | None = None,
    resume_from_manifest: str | None = None,
) -> object:
    return OmegaConf.create(
        {
            "source": str(images_root),
            "masks_root": str(masks_root) if masks_root is not None else None,
            "destination": str(destination),
            "artifact_dir": str(artifact_dir),
            "iqr_columns": [],
            "iqr_factor": 1.5,
            "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
            "workers": 1,
            "storage_options": None,
            "build_shards": False,
            "shard_root": None,
            "shard_size": 1000,
            "run_label": run_label,
            "resume_from_parquet": resume_from_parquet,
            "resume_from_filtered": resume_from_filtered,
            "resume_from_split": resume_from_split,
            "resume_from_manifest": resume_from_manifest,
            "haralick": {"features": ["contrast"], "distances": None, "angles": None},
        }
    )


def _read_manifest(path: str) -> list[dict]:
    with open(path, "rt", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ---------------------------------------------------------------------------
# Scenario 1: one record per input image
# ---------------------------------------------------------------------------


def test_pipeline_produces_exactly_one_manifest_record_per_input_image(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg = _minimal_cfg(images, tmp_path / "out", tmp_path / "artifacts", run_label="s1")
    manifest_path = etl_flow(cfg)
    records = _read_manifest(manifest_path)
    assert len(records) == 4


# ---------------------------------------------------------------------------
# Scenario 2: no masks -> lung_out_of_frame is null
# ---------------------------------------------------------------------------


def test_images_processed_without_masks_have_null_lung_out_of_frame_in_manifest(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg = _minimal_cfg(
        images,
        tmp_path / "out",
        tmp_path / "artifacts",
        masks_root=None,
        run_label="s2",
    )
    manifest_path = etl_flow(cfg)
    records = _read_manifest(manifest_path)
    assert all(r["lung_out_of_frame"] is None for r in records)


# ---------------------------------------------------------------------------
# Scenario 3: border-touching masks -> records present but flagged excluded
# ---------------------------------------------------------------------------


def test_images_with_border_touching_masks_are_flagged_excluded_in_manifest(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    masks = _build_border_mask_tree(tmp_path, images)
    cfg = _minimal_cfg(
        images,
        tmp_path / "out",
        tmp_path / "artifacts",
        masks_root=masks,
        run_label="s3",
    )
    manifest_path = etl_flow(cfg)
    records = _read_manifest(manifest_path)
    assert len(records) == 4
    assert any(r["excluded"] for r in records)
    assert any(r["lung_out_of_frame"] is True for r in records)


# ---------------------------------------------------------------------------
# Scenario 4: every record has a valid non-empty split
# ---------------------------------------------------------------------------


def test_every_manifest_record_has_a_valid_non_empty_split_assignment(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg = _minimal_cfg(images, tmp_path / "out", tmp_path / "artifacts", run_label="s4")
    manifest_path = etl_flow(cfg)
    records = _read_manifest(manifest_path)
    valid_splits = {"train", "val", "test"}
    for rec in records:
        assert rec["split"] in valid_splits, f"unexpected split: {rec['split']!r}"


# ---------------------------------------------------------------------------
# Scenario 5: all records in a run share the same manifest_id
# ---------------------------------------------------------------------------


def test_all_records_in_the_same_run_share_the_same_manifest_id(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg = _minimal_cfg(images, tmp_path / "out", tmp_path / "artifacts", run_label="s5")
    manifest_path = etl_flow(cfg)
    records = _read_manifest(manifest_path)
    ids = {r["manifest_id"] for r in records}
    assert len(ids) == 1


# ---------------------------------------------------------------------------
# Scenario 6: idempotency — same config twice yields same path and record count
# ---------------------------------------------------------------------------


def test_running_pipeline_twice_with_identical_config_is_idempotent(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg = _minimal_cfg(images, tmp_path / "out", tmp_path / "artifacts", run_label="s6")
    path1 = etl_flow(cfg)
    path2 = etl_flow(cfg)
    assert path1 == path2
    assert len(_read_manifest(path1)) == len(_read_manifest(path2))


# ---------------------------------------------------------------------------
# Scenario 7: run_label changes the run ID and is stable across identical calls
# ---------------------------------------------------------------------------


def test_compute_run_id_with_run_label_differs_from_no_label_and_is_stable(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg_no_label = OmegaConf.create({"source": str(images), "run_label": None})
    cfg_with_label = OmegaConf.create({"source": str(images), "run_label": "my-label"})

    id_no_label = compute_run_id(cfg_no_label, str(images))
    id_with_label_1 = compute_run_id(cfg_with_label, str(images))
    id_with_label_2 = compute_run_id(cfg_with_label, str(images))

    assert id_with_label_1 != id_no_label
    assert id_with_label_1 == id_with_label_2


# ---------------------------------------------------------------------------
# Scenario 8: resume_from_manifest skips re-processing
# ---------------------------------------------------------------------------


def test_passing_resume_manifest_path_skips_re_processing_and_returns_consistent_manifest(
    tmp_path: Path,
) -> None:
    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg_first = _minimal_cfg(
        images, tmp_path / "out", tmp_path / "artifacts", run_label="s8"
    )
    first_manifest = etl_flow(cfg_first)
    first_records = _read_manifest(first_manifest)

    cfg_resume = _minimal_cfg(
        images,
        tmp_path / "out",
        tmp_path / "artifacts",
        run_label="s8",
        resume_from_manifest=first_manifest,
    )
    resumed_manifest = etl_flow(cfg_resume)
    resumed_records = _read_manifest(resumed_manifest)

    assert len(resumed_records) == len(first_records)
    assert resumed_manifest == first_manifest


# ---------------------------------------------------------------------------
# Fix 1: _build_shards portable core has no Prefect imports
# ---------------------------------------------------------------------------


def test_build_shards_portable_core_contains_no_prefect_imports() -> None:
    assert "prefect" not in inspect.getsource(_build_shards)


# ---------------------------------------------------------------------------
# Fix 2: main entry point is importable and callable
# ---------------------------------------------------------------------------


def test_main_entry_point_is_callable() -> None:
    assert callable(main)


# ---------------------------------------------------------------------------
# FIX C1: pipeline module exposes compute_run_id even when Prefect is absent
# ---------------------------------------------------------------------------


def test_pipeline_module_exposes_compute_run_id_without_prefect_at_call_time() -> None:
    from radiologist.etl.pipeline import (  # noqa: F401 — import is the test
        compute_run_id,
    )

    assert callable(compute_run_id)


# ---------------------------------------------------------------------------
# FIX C2: build_shards_task accepts storage_options kwarg
# ---------------------------------------------------------------------------

_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def test_build_shards_task_accepts_storage_options(tmp_path: Path) -> None:
    from radiologist.etl.pipeline import build_shards_task

    images = _build_image_tree(tmp_path, n_per_class=2)
    cfg = _minimal_cfg(images, tmp_path / "out", tmp_path / "artifacts", run_label="c2")
    manifest_path = etl_flow(cfg)

    result = build_shards_task(
        manifest_path,
        str(tmp_path / "shards"),
        _RATIOS,
        storage_options={},
    )
    assert result == manifest_path
