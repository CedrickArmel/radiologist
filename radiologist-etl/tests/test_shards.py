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
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from radiologist.etl.manifest import JsonlWriter, ManifestRecord
from radiologist.etl.shards import build_shards

_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _make_png(tmp_path, name, label) -> str:
    p = tmp_path / label / name
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(str(p))
    return str(p)


def _make_manifest(tmp_path, records):
    path = str(tmp_path / "manifest-abc123def456789.jsonl")
    JsonlWriter().write(records, path)
    return path


def _read_tar_members(tar_path: str) -> list[str]:
    with tarfile.open(tar_path) as tf:
        return [m.name for m in tf.getmembers()]


def _read_tar_bytes(tar_path: str, member_name: str) -> bytes:
    with tarfile.open(tar_path) as tf:
        f = tf.extractfile(member_name)
        assert f is not None
        return f.read()


class TestBuildShards:
    def test_shard_field_populated_on_non_excluded_and_null_on_excluded(self, tmp_path):
        path_1 = _make_png(tmp_path, "normal_001.png", "NORMAL")
        path_2 = _make_png(tmp_path, "normal_002.png", "NORMAL")
        path_excl = _make_png(tmp_path, "normal_excl.png", "NORMAL")

        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_1,
                filename="normal_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_2,
                filename="normal_002.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_excl,
                filename="normal_excl.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=True,
                exclusion_reason="iqr:contrast",
            ),
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        build_shards(manifest_path, shard_root, _RATIOS)

        lines = Path(manifest_path).read_text().splitlines()
        parsed = [json.loads(line) for line in lines]
        non_excluded = [p for p in parsed if not p["excluded"]]
        excluded = [p for p in parsed if p["excluded"]]

        assert len(non_excluded) == 2
        assert all(p["shard"] is not None for p in non_excluded)
        assert excluded[0]["shard"] is None

    def test_tar_contains_exactly_non_excluded_samples(self, tmp_path):
        path_1 = _make_png(tmp_path, "normal_001.png", "NORMAL")
        path_2 = _make_png(tmp_path, "normal_002.png", "NORMAL")
        path_excl = _make_png(tmp_path, "normal_excl.png", "NORMAL")

        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_1,
                filename="normal_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_2,
                filename="normal_002.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_excl,
                filename="normal_excl.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=True,
                exclusion_reason="iqr:contrast",
            ),
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        build_shards(manifest_path, shard_root, _RATIOS)

        shard_path = str(
            tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000000.tar"
        )
        members = _read_tar_members(shard_path)
        png_members = [m for m in members if m.endswith(".png")]
        assert len(png_members) == 2

    def test_tar_member_keys_match_image_stems(self, tmp_path):
        path_a = _make_png(tmp_path, "xray_a.png", "NORMAL")
        path_b = _make_png(tmp_path, "xray_b.png", "NORMAL")

        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_a,
                filename="xray_a.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_b,
                filename="xray_b.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        build_shards(manifest_path, shard_root, _RATIOS)

        shard_path = str(
            tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000000.tar"
        )
        members = _read_tar_members(shard_path)
        png_keys = {m.rsplit(".", 1)[0] for m in members if m.endswith(".png")}
        assert "xray_a" in png_keys
        assert "xray_b" in png_keys

    def test_cls_bytes_decode_to_label_string(self, tmp_path):
        path_img = _make_png(tmp_path, "img_001.png", "NORMAL")

        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_img,
                filename="img_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        build_shards(manifest_path, shard_root, _RATIOS)

        shard_path = str(
            tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000000.tar"
        )
        cls_bytes = _read_tar_bytes(shard_path, "img_001.cls")
        assert cls_bytes.decode("utf-8") == "NORMAL"

    def test_excluded_records_absent_from_shard_tar(self, tmp_path):
        path_normal = _make_png(tmp_path, "normal_001.png", "NORMAL")
        path_excl = _make_png(tmp_path, "excluded_001.png", "NORMAL")

        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_normal,
                filename="normal_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_excl,
                filename="excluded_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=True,
                exclusion_reason="iqr:contrast",
            ),
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        build_shards(manifest_path, shard_root, _RATIOS)

        shard_path = str(
            tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000000.tar"
        )
        members = _read_tar_members(shard_path)
        assert not any("excluded_001" in m for m in members)

    def test_split_report_written_next_to_manifest_with_correct_structure(
        self, tmp_path
    ):
        path_img = _make_png(tmp_path, "img_001.png", "NORMAL")

        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_img,
                filename="img_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        build_shards(manifest_path, shard_root, _RATIOS)

        report_path = tmp_path / "split-report-abc123def456789.json"
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["run_id"] == "abc123def456789"
        assert "configured_ratios" in report
        assert "observed" in report

    def test_start_shard_index_offsets_shard_filename(self, tmp_path):
        path_img = _make_png(tmp_path, "img_001.png", "NORMAL")

        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_img,
                filename="img_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        build_shards(
            manifest_path,
            shard_root,
            _RATIOS,
            start_shard_index={("train", "NORMAL"): 5},
        )

        expected_shard = (
            tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000005.tar"
        )
        assert expected_shard.exists()

    def test_returns_same_manifest_path_as_input(self, tmp_path):
        path_img = _make_png(tmp_path, "img_001.png", "NORMAL")

        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_img,
                filename="img_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        result = build_shards(manifest_path, shard_root, _RATIOS)

        assert result == manifest_path

    def test_manifest_rewritten_with_shard_field_on_non_excluded_after_build(
        self, tmp_path
    ):
        path_normal = _make_png(tmp_path, "normal_001.png", "NORMAL")
        path_excl = _make_png(tmp_path, "excl_001.png", "NORMAL")

        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_normal,
                filename="normal_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            ),
            ManifestRecord(
                manifest_id="abc123def456789",
                path=path_excl,
                filename="excl_001.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=True,
                exclusion_reason="iqr:contrast",
            ),
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        build_shards(manifest_path, shard_root, _RATIOS)

        lines = Path(manifest_path).read_text().splitlines()
        parsed = [json.loads(line) for line in lines]
        non_excluded = [p for p in parsed if not p["excluded"]]
        excluded_recs = [p for p in parsed if p["excluded"]]

        assert all(p["shard"] is not None for p in non_excluded)
        assert excluded_recs[0]["shard"] is None

    def test_shard_size_batching_creates_multiple_shards_for_five_records(
        self, tmp_path
    ):
        """5 non-excluded records with shard_size=2 must produce 3 tar files (2+2+1)."""
        paths = [_make_png(tmp_path, f"normal_{i:03d}.png", "NORMAL") for i in range(5)]
        records = [
            ManifestRecord(
                manifest_id="abc123def456789",
                path=paths[i],
                filename=f"normal_{i:03d}.png",
                label="NORMAL",
                split="train",
                stats={},
                excluded=False,
            )
            for i in range(5)
        ]
        manifest_path = _make_manifest(tmp_path, records)
        shard_root = str(tmp_path / "shards")

        build_shards(manifest_path, shard_root, _RATIOS, shard_size=2)

        shard_dir = tmp_path / "shards" / "train" / "NORMAL"
        tar_files = sorted(shard_dir.glob("*.tar"))
        assert len(tar_files) == 3

        lines = Path(manifest_path).read_text().splitlines()
        parsed = [json.loads(line) for line in lines]
        assert all(p["shard"] is not None for p in parsed)
