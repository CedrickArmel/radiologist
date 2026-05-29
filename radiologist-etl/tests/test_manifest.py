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

import pandas as pd
import pytest

from radiologist.etl.manifest import JsonlWriter, ManifestRecord, ParquetWriter


def _make_record(
    manifest_id: str = "abcd1234efgh5678",
    path: str = "gs://bucket/scan.png",
    filename: str = "scan.png",
    label: str = "normal",
    split: str = "train",
    stats: dict | None = None,
    lung_out_of_frame: bool | None = None,
    excluded: bool = False,
    exclusion_reason: str = "",
    shard: str | None = None,
) -> ManifestRecord:
    if stats is None:
        stats = {"haralick_contrast": 0.5, "asymmetry_score": 0.1}
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


class TestParquetWriter:
    def test_write_preserves_all_field_values(self, tmp_path):
        record = _make_record(
            manifest_id="abcd1234efgh5678",
            path="gs://bucket/scan.png",
            filename="scan.png",
            label="normal",
            split="train",
            stats={"haralick_contrast": 0.5, "asymmetry_score": 0.1},
            lung_out_of_frame=True,
            excluded=False,
            exclusion_reason="",
            shard="shard-001",
        )
        dest = str(tmp_path / "manifest.parquet")

        ParquetWriter().write([record], dest)
        df = pd.read_parquet(dest)

        assert df.iloc[0]["manifest_id"] == "abcd1234efgh5678"
        assert df.iloc[0]["path"] == "gs://bucket/scan.png"
        assert df.iloc[0]["filename"] == "scan.png"
        assert df.iloc[0]["label"] == "normal"
        assert df.iloc[0]["split"] == "train"
        assert df.iloc[0]["haralick_contrast"] == pytest.approx(0.5)
        assert df.iloc[0]["asymmetry_score"] == pytest.approx(0.1)
        assert df.iloc[0]["lung_out_of_frame"] is True
        assert df.iloc[0]["excluded"] is False
        assert df.iloc[0]["exclusion_reason"] == ""
        assert df.iloc[0]["shard"] == "shard-001"

    def test_write_lung_out_of_frame_none_reads_as_pd_na(self, tmp_path):
        record = _make_record(lung_out_of_frame=None)
        dest = str(tmp_path / "manifest.parquet")

        ParquetWriter().write([record], dest)
        df = pd.read_parquet(dest)

        assert pd.isna(df.iloc[0]["lung_out_of_frame"])

    def test_write_shard_none_reads_as_pd_na(self, tmp_path):
        record = _make_record(shard=None)
        dest = str(tmp_path / "manifest.parquet")

        ParquetWriter().write([record], dest)
        df = pd.read_parquet(dest)

        assert pd.isna(df.iloc[0]["shard"])

    def test_write_stats_are_flattened_to_top_level_columns(self, tmp_path):
        record = _make_record(stats={"haralick_contrast": 1.2, "haralick_energy": 0.8})
        dest = str(tmp_path / "manifest.parquet")

        ParquetWriter().write([record], dest)
        df = pd.read_parquet(dest)

        assert "haralick_contrast" in df.columns
        assert "haralick_energy" in df.columns
        assert "stats" not in df.columns

    def test_write_works_with_local_path(self, tmp_path):
        record = _make_record()
        dest = str(tmp_path / "manifest.parquet")

        ParquetWriter().write([record], dest)

        assert (tmp_path / "manifest.parquet").exists()

    def test_write_multiple_records(self, tmp_path):
        records = [_make_record(manifest_id=f"id{i:014d}") for i in range(5)]
        dest = str(tmp_path / "manifest.parquet")

        ParquetWriter().write(records, dest)
        df = pd.read_parquet(dest)

        assert len(df) == 5


class TestJsonlWriter:
    def test_write_produces_one_line_per_record_when_all_included(self, tmp_path):
        records = [_make_record() for _ in range(3)]
        dest = str(tmp_path / "manifest.jsonl")

        JsonlWriter().write(records, dest)

        lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
        assert len(lines) == 3

    def test_write_produces_one_line_per_record_regardless_of_excluded_flag(
        self, tmp_path
    ):
        records = [
            _make_record(excluded=True, exclusion_reason="iqr:haralick_contrast"),
            _make_record(excluded=False),
        ]
        dest = str(tmp_path / "manifest.jsonl")

        JsonlWriter().write(records, dest)

        lines = (tmp_path / "manifest.jsonl").read_text().splitlines()
        assert len(lines) == 2

    def test_every_jsonl_line_contains_manifest_id(self, tmp_path):
        records = [_make_record(manifest_id="abcd1234efgh5678") for _ in range(3)]
        dest = str(tmp_path / "manifest.jsonl")

        JsonlWriter().write(records, dest)

        for line in (tmp_path / "manifest.jsonl").read_text().splitlines():
            obj = json.loads(line)
            assert obj["manifest_id"] == "abcd1234efgh5678"

    def test_shard_none_serialises_as_json_null(self, tmp_path):
        record = _make_record(shard=None)
        dest = str(tmp_path / "manifest.jsonl")

        JsonlWriter().write([record], dest)

        line = (tmp_path / "manifest.jsonl").read_text().strip()
        obj = json.loads(line)
        assert obj["shard"] is None

    def test_stats_are_flattened_in_jsonl_output(self, tmp_path):
        record = _make_record(stats={"haralick_contrast": 0.5, "asymmetry_score": 0.1})
        dest = str(tmp_path / "manifest.jsonl")

        JsonlWriter().write([record], dest)

        obj = json.loads((tmp_path / "manifest.jsonl").read_text().strip())
        assert "haralick_contrast" in obj
        assert "asymmetry_score" in obj
        assert "stats" not in obj

    def test_write_works_with_local_path(self, tmp_path):
        record = _make_record()
        dest = str(tmp_path / "manifest.jsonl")

        JsonlWriter().write([record], dest)

        assert (tmp_path / "manifest.jsonl").exists()
