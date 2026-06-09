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

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from radiologist.etl import JsonlWriter, ManifestRecord, ParquetWriter


def _record(
    manifest_id: str = "run-abc-0000001",
    path: str = "/data/NORMAL/scan.png",
    filename: str = "scan.png",
    label: str = "NORMAL",
    split: str = "train",
    stats: dict | None = None,
    lung_out_of_frame: bool | None = None,
    excluded: bool = False,
    exclusion_reason: str = "",
    shard: str | None = None,
) -> ManifestRecord:
    if stats is None:
        stats = {"haralick_contrast": 0.42, "asymmetry_ratio": 1.05}
    return ManifestRecord(
        manifest_id=manifest_id,
        path=path,
        filename=filename,
        label=label,
        split=split,
        stats=stats,
        lung_out_of_frame=lung_out_of_frame,
        excluded=excluded,
        exclusion_reason=exclusion_reason,
        shard=shard,
    )


# --- Parquet round-trip ---


def test_all_scalar_fields_round_trip_through_parquet_unchanged(tmp_path: Path) -> None:
    rec = _record(
        manifest_id="run-abc-0000001",
        path="/data/NORMAL/scan.png",
        filename="scan.png",
        label="NORMAL",
        split="val",
        exclusion_reason="iqr:haralick_contrast",
        shard="train/NORMAL/train-normal-000000.tar",
    )
    dest = str(tmp_path / "manifest.parquet")
    ParquetWriter().write([rec], dest)
    df = pd.read_parquet(dest)
    row = df.iloc[0]
    assert row["manifest_id"] == "run-abc-0000001"
    assert row["path"] == "/data/NORMAL/scan.png"
    assert row["filename"] == "scan.png"
    assert row["label"] == "NORMAL"
    assert row["split"] == "val"
    assert row["exclusion_reason"] == "iqr:haralick_contrast"
    assert row["shard"] == "train/NORMAL/train-normal-000000.tar"


def test_stat_features_round_trip_as_float64_columns_in_parquet(tmp_path: Path) -> None:
    rec = _record(stats={"haralick_contrast": 0.42, "asymmetry_ratio": 1.05})
    dest = str(tmp_path / "manifest.parquet")
    ParquetWriter().write([rec], dest)
    df = pd.read_parquet(dest)
    assert "haralick_contrast" in df.columns
    assert "asymmetry_ratio" in df.columns
    assert "stats" not in df.columns
    assert df["haralick_contrast"].dtype == "float64"
    assert df["asymmetry_ratio"].dtype == "float64"
    assert df.iloc[0]["haralick_contrast"] == pytest.approx(0.42)


def test_lung_out_of_frame_none_round_trips_as_null_in_parquet(tmp_path: Path) -> None:
    rec = _record(lung_out_of_frame=None)
    dest = str(tmp_path / "manifest.parquet")
    ParquetWriter().write([rec], dest)
    df = pd.read_parquet(dest)
    assert pd.isna(df.iloc[0]["lung_out_of_frame"])


# --- JSONL round-trip ---


def test_shard_none_round_trips_as_json_null_in_jsonl_output(tmp_path: Path) -> None:
    rec = _record(shard=None)
    dest = str(tmp_path / "manifest.jsonl")
    JsonlWriter().write([rec], dest)
    obj = json.loads(Path(dest).read_text().strip())
    assert obj["shard"] is None


def test_jsonl_output_contains_exactly_one_line_per_record_regardless_of_excluded_flag(
    tmp_path: Path,
) -> None:
    records = [
        _record(excluded=True, exclusion_reason="iqr:haralick_contrast"),
        _record(excluded=False),
        _record(excluded=True, exclusion_reason="lung_out_of_frame"),
    ]
    dest = str(tmp_path / "manifest.jsonl")
    JsonlWriter().write(records, dest)
    lines = [ln for ln in Path(dest).read_text().splitlines() if ln.strip()]
    assert len(lines) == 3


def test_every_jsonl_line_contains_a_manifest_id_field(tmp_path: Path) -> None:
    records = [_record(manifest_id="run-xyz-001") for _ in range(3)]
    dest = str(tmp_path / "manifest.jsonl")
    JsonlWriter().write(records, dest)
    for line in Path(dest).read_text().splitlines():
        if line.strip():
            assert "manifest_id" in json.loads(line)


def test_stat_features_are_inlined_at_top_level_of_each_jsonl_object(
    tmp_path: Path,
) -> None:
    rec = _record(stats={"haralick_contrast": 0.7, "asymmetry_ratio": 1.2})
    dest = str(tmp_path / "manifest.jsonl")
    JsonlWriter().write([rec], dest)
    obj = json.loads(Path(dest).read_text().strip())
    assert "haralick_contrast" in obj
    assert "asymmetry_ratio" in obj
    assert "stats" not in obj


# ---------------------------------------------------------------------------
# FIX C3: ParquetWriter raises ValueError on empty records list
# ---------------------------------------------------------------------------


def test_parquet_writer_raises_on_empty_records(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        ParquetWriter().write([], str(tmp_path / "empty.parquet"))
