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

import json
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image

from radiologist.etl.manifest import ManifestRecord, ParquetWriter
from radiologist.etl.pipeline import (
    _apply_filters,
    _assign_splits,
    _compute_stats,
    _df_to_records,
    _write_jsonl,
    compute_run_id,
    etl_flow,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_png(path: Path) -> None:
    """Save a 10x10 random RGB PNG to path."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _make_interior_mask_png(path: Path) -> None:
    """Save a 10x10 mask with nonzero only in the interior."""
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    arr[3:7, 3:7] = 255
    Image.fromarray(arr).save(path)


def _build_two_class_tree(base: Path, n_per_class: int = 2) -> Path:
    """Create images/NORMAL/ and images/ABNORMAL/ each with n images."""
    images_root = base / "images"
    for cls in ("NORMAL", "ABNORMAL"):
        (images_root / cls).mkdir(parents=True)
        for i in range(n_per_class):
            _make_rgb_png(images_root / cls / f"img{i:03d}.png")
    return images_root


def _minimal_parquet_with_records(
    path: Path,
    n_records: int = 3,
    *,
    include_outlier: bool = False,
) -> Path:
    """Write a minimal parquet file with n ManifestRecord-shaped rows."""
    rows = []
    for i in range(n_records):
        rows.append(
            {
                "manifest_id": "test-run-001",
                "path": f"/data/NORMAL/img{i:03d}.png",
                "filename": f"img{i:03d}.png",
                "label": "NORMAL",
                "split": "",
                "shard": None,
                "haralick_contrast": float(i + 1),
                "lung_out_of_frame": None,
                "excluded": False,
                "exclusion_reason": "",
            }
        )
    if include_outlier:
        rows.append(
            {
                "manifest_id": "test-run-001",
                "path": "/data/NORMAL/outlier.png",
                "filename": "outlier.png",
                "label": "NORMAL",
                "split": "",
                "shard": None,
                "haralick_contrast": 9999.0,
                "lung_out_of_frame": None,
                "excluded": False,
                "exclusion_reason": "",
            }
        )
    df = pd.DataFrame(rows)
    dest = path / "stats-test-run-001.parquet"
    df.to_parquet(dest)
    return dest


# ---------------------------------------------------------------------------
# AC1: compute_run_id is idempotent (same result on repeated calls)
# ---------------------------------------------------------------------------


def test_compute_run_id_returns_same_value_on_repeated_calls(
    tmp_path: Path,
) -> None:
    images_root = _build_two_class_tree(tmp_path)
    cfg = OmegaConf.create(
        {
            "source": str(images_root),
            "run_label": None,
            "destination": str(tmp_path / "out"),
            "artifact_dir": str(tmp_path / "artifacts"),
        }
    )
    id1 = compute_run_id(cfg, str(images_root))
    id2 = compute_run_id(cfg, str(images_root))
    assert id1 == id2


# ---------------------------------------------------------------------------
# AC2: compute_run_id returns different value when run_label set vs not set
# ---------------------------------------------------------------------------


def test_compute_run_id_differs_when_run_label_set_vs_not_set(
    tmp_path: Path,
) -> None:
    images_root = _build_two_class_tree(tmp_path)
    cfg_no_label = OmegaConf.create(
        {
            "source": str(images_root),
            "run_label": None,
        }
    )
    cfg_with_label = OmegaConf.create(
        {
            "source": str(images_root),
            "run_label": "my-custom-label",
        }
    )
    id_no_label = compute_run_id(cfg_no_label, str(images_root))
    id_with_label = compute_run_id(cfg_with_label, str(images_root))
    assert id_no_label != id_with_label


# ---------------------------------------------------------------------------
# AC3: compute_run_id returns run_label directly when set
# ---------------------------------------------------------------------------


def test_compute_run_id_returns_run_label_directly_when_set(
    tmp_path: Path,
) -> None:
    images_root = _build_two_class_tree(tmp_path)
    cfg = OmegaConf.create(
        {
            "source": str(images_root),
            "run_label": "explicit-run-id-42",
        }
    )
    result = compute_run_id(cfg, str(images_root))
    assert result == "explicit-run-id-42"


# ---------------------------------------------------------------------------
# AC4: _compute_stats produces parquet with 4 rows, each with correct manifest_id
# ---------------------------------------------------------------------------


def test_compute_stats_produces_parquet_with_correct_rows(tmp_path: Path) -> None:
    images_root = _build_two_class_tree(tmp_path, n_per_class=2)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    from radiologist.etl.stats import make_haralick

    extractors = [make_haralick(features=["contrast"])]
    parquet_path = _compute_stats(
        source=str(images_root),
        run_id="run-abc-0001",
        masks_root=None,
        extractors=extractors,
        workers=1,
        artifact_dir=str(artifact_dir),
    )

    assert parquet_path.endswith(".parquet")
    assert Path(parquet_path).exists()
    df = pd.read_parquet(parquet_path)
    assert len(df) == 4
    assert (df["manifest_id"] == "run-abc-0001").all()


# ---------------------------------------------------------------------------
# AC5: _apply_filters sets excluded=True on outlier rows
# ---------------------------------------------------------------------------


def test_apply_filters_marks_outliers_as_excluded(tmp_path: Path) -> None:
    parquet_path = _minimal_parquet_with_records(
        tmp_path, n_records=5, include_outlier=True
    )

    filtered_path = _apply_filters(
        parquet_path=str(parquet_path),
        iqr_columns=["haralick_contrast"],
        factor=1.5,
    )

    df = pd.read_parquet(filtered_path)
    outlier_rows = df[df["haralick_contrast"] > 100.0]
    assert len(outlier_rows) == 1
    assert bool(outlier_rows.iloc[0]["excluded"]) is True


# ---------------------------------------------------------------------------
# AC6: _assign_splits assigns non-empty split values from ratio keys
# ---------------------------------------------------------------------------


def test_assign_splits_assigns_non_empty_split_values(tmp_path: Path) -> None:
    parquet_path = _minimal_parquet_with_records(tmp_path, n_records=10)
    filtered_path = str(parquet_path).replace(".parquet", "-filtered.parquet")
    pd.read_parquet(parquet_path).to_parquet(filtered_path)

    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    split_path = _assign_splits(
        parquet_path=filtered_path,
        ratios=ratios,
    )

    df = pd.read_parquet(split_path)
    valid_splits = set(ratios.keys())
    assert all(s in valid_splits for s in df["split"].tolist())
    assert all(s != "" for s in df["split"].tolist())


# ---------------------------------------------------------------------------
# AC7: _write_jsonl produces a JSONL file with exactly 3 lines
# ---------------------------------------------------------------------------


def test_write_jsonl_produces_correct_line_count(tmp_path: Path) -> None:
    rows = [
        {
            "manifest_id": "run-001",
            "path": f"/data/NORMAL/img{i:03d}.png",
            "filename": f"img{i:03d}.png",
            "label": "NORMAL",
            "split": "train",
            "shard": None,
            "haralick_contrast": float(i + 1),
            "lung_out_of_frame": None,
            "excluded": False,
            "exclusion_reason": "",
        }
        for i in range(3)
    ]
    split_parquet = tmp_path / "stats-run-001-split.parquet"
    pd.DataFrame(rows).to_parquet(split_parquet)

    destination = str(tmp_path / "manifest-run-001.jsonl")
    result_path = _write_jsonl(
        parquet_path=str(split_parquet),
        destination=destination,
    )

    assert result_path == destination
    with open(destination, "rt", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) == 3


# ---------------------------------------------------------------------------
# AC8: Integration — _compute_stats → _apply_filters → _assign_splits
#       → _write_jsonl with 5-image fixture (3 with masks, 2 without)
# ---------------------------------------------------------------------------


def test_integration_pipeline_produces_correct_manifest(tmp_path: Path) -> None:
    images_root = tmp_path / "images"
    masks_root = tmp_path / "masks"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    (images_root / "NORMAL").mkdir(parents=True)
    (images_root / "ABNORMAL").mkdir(parents=True)
    (masks_root / "NORMAL").mkdir(parents=True)

    for i in range(3):
        _make_rgb_png(images_root / "NORMAL" / f"img{i:03d}.png")
        _make_interior_mask_png(masks_root / "NORMAL" / f"img{i:03d}.png")
    for i in range(2):
        _make_rgb_png(images_root / "ABNORMAL" / f"img{i:03d}.png")

    from radiologist.etl.stats import make_haralick

    run_id = "integration-test-01"
    extractors = [make_haralick(features=["contrast"])]

    parquet_path = _compute_stats(
        source=str(images_root),
        run_id=run_id,
        masks_root=str(masks_root),
        extractors=extractors,
        workers=1,
        artifact_dir=str(artifact_dir),
    )
    filtered_path = _apply_filters(
        parquet_path=parquet_path,
        iqr_columns=[],
        factor=1.5,
    )
    split_path = _assign_splits(
        parquet_path=filtered_path,
        ratios={"train": 0.70, "val": 0.15, "test": 0.15},
    )
    destination = str(tmp_path / f"manifest-{run_id}.jsonl")
    manifest_path = _write_jsonl(
        parquet_path=split_path,
        destination=destination,
    )

    with open(manifest_path, "rt", encoding="utf-8") as f:
        records = [json.loads(ln) for ln in f if ln.strip()]
    assert len(records) == 5
    assert all(r["split"] in {"train", "val", "test"} for r in records)

    normal_records = [r for r in records if r["label"] == "NORMAL"]
    abnormal_records = [r for r in records if r["label"] == "ABNORMAL"]
    assert all(r["lung_out_of_frame"] is not None for r in normal_records)
    assert all(r["lung_out_of_frame"] is None for r in abnormal_records)
    assert all(r["manifest_id"] == run_id for r in records)


# ---------------------------------------------------------------------------
# AC9: etl_flow called twice with same config returns same manifest path
#      and same line count (idempotent via Prefect task caching)
# ---------------------------------------------------------------------------


def test_etl_flow_is_idempotent_for_identical_config(tmp_path: Path) -> None:
    images_root = _build_two_class_tree(tmp_path, n_per_class=2)
    artifact_dir = tmp_path / "artifacts"
    destination = tmp_path / "out"
    artifact_dir.mkdir()
    destination.mkdir()

    cfg = OmegaConf.create(
        {
            "source": str(images_root),
            "masks_root": None,
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
            "run_label": "idempotency-test",
            "resume_from_parquet": None,
            "resume_from_filtered": None,
            "resume_from_split": None,
            "resume_from_manifest": None,
            "haralick": {"features": ["contrast"], "distances": None, "angles": None},
        }
    )

    path1 = etl_flow(cfg)
    path2 = etl_flow(cfg)

    assert path1 == path2

    with open(path1, "rt", encoding="utf-8") as f:
        lines1 = [ln for ln in f if ln.strip()]
    with open(path2, "rt", encoding="utf-8") as f:
        lines2 = [ln for ln in f if ln.strip()]
    assert len(lines1) == len(lines2)


# ---------------------------------------------------------------------------
# FIX 2: pd.NA shard detection — shard must be None, not "<NA>" string
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FIX 4: _apply_filters / _assign_splits use fsspec-backed writes + path fix
# ---------------------------------------------------------------------------


def test_apply_filters_accepts_storage_options_kwarg(tmp_path: Path) -> None:
    """_apply_filters with storage_options={} must not crash."""
    parquet_path = _minimal_parquet_with_records(tmp_path, n_records=3)
    filtered_path = _apply_filters(
        parquet_path=str(parquet_path),
        iqr_columns=[],
        factor=1.5,
        storage_options={},
    )
    assert Path(filtered_path).exists()


def test_assign_splits_accepts_storage_options_kwarg(tmp_path: Path) -> None:
    """_assign_splits with storage_options={} must not crash."""
    parquet_path = _minimal_parquet_with_records(tmp_path, n_records=5)
    filtered_path = str(parquet_path).replace(".parquet", "-filtered.parquet")
    pd.read_parquet(parquet_path).to_parquet(filtered_path)
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    split_path = _assign_splits(
        parquet_path=filtered_path,
        ratios=ratios,
        storage_options={},
    )
    assert Path(split_path).exists()


def test_apply_filters_output_path_does_not_double_append_filtered(
    tmp_path: Path,
) -> None:
    """Output path must end with -filtered.parquet, not -filtered-filtered.parquet."""
    parquet_path = _minimal_parquet_with_records(tmp_path, n_records=3)
    filtered_path = _apply_filters(
        parquet_path=str(parquet_path),
        iqr_columns=[],
        factor=1.5,
    )
    assert filtered_path.endswith("-filtered.parquet")
    assert "-filtered-filtered" not in filtered_path


def test_assign_splits_output_path_stem_correct(tmp_path: Path) -> None:
    """Output path must end with -split.parquet derived from stats-{run_id}."""
    parquet_path = _minimal_parquet_with_records(tmp_path, n_records=5)
    filtered_path = str(parquet_path).replace(".parquet", "-filtered.parquet")
    pd.read_parquet(parquet_path).to_parquet(filtered_path)
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    split_path = _assign_splits(parquet_path=filtered_path, ratios=ratios)
    assert split_path.endswith("-split.parquet")
    assert "filtered" not in Path(split_path).stem


# ---------------------------------------------------------------------------
# FIX 2: pd.NA shard detection — shard must be None, not "<NA>" string
# ---------------------------------------------------------------------------


def test_df_to_records_shard_is_none_when_parquet_null_via_parquet_writer(
    tmp_path: Path,
) -> None:
    """Records written with shard=None via ParquetWriter must come back as None."""
    record = ManifestRecord(
        manifest_id="fix2-run-001",
        path="/data/NORMAL/img001.png",
        filename="img001.png",
        label="NORMAL",
        split="train",
        stats={"haralick_contrast": 1.0},
        shard=None,
    )
    dest = str(tmp_path / "stats-fix2-run-001.parquet")
    ParquetWriter().write([record], dest)

    df = pd.read_parquet(dest)
    records = _df_to_records(df)

    assert len(records) == 1
    assert records[0].shard is None, f"Expected None, got {records[0].shard!r}"


def test_df_to_records_shard_is_none_when_column_uses_pandas_string_dtype(
    tmp_path: Path,
) -> None:
    """When shard column is StringDtype (pd.NA on null), must still return None."""
    rows = [
        {
            "manifest_id": "fix2-run-002",
            "path": "/data/NORMAL/img001.png",
            "filename": "img001.png",
            "label": "NORMAL",
            "split": "train",
            "shard": pd.NA,
            "haralick_contrast": 1.0,
            "lung_out_of_frame": None,
            "excluded": False,
            "exclusion_reason": "",
        }
    ]
    df = pd.DataFrame(rows)
    df["shard"] = df["shard"].astype(pd.StringDtype())

    records = _df_to_records(df)

    assert len(records) == 1
    assert records[0].shard is None, f"Expected None, got {records[0].shard!r}"
