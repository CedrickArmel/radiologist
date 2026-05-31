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

from radiologist.etl import JsonlWriter, ManifestRecord, build_shards

_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def _make_png(root: Path, filename: str, label: str) -> str:
    dest = root / label / filename
    _write_png(dest, np.zeros((10, 10, 3), dtype=np.uint8))
    return str(dest)


def _make_manifest(root: Path, records: list[ManifestRecord]) -> str:
    path = str(root / "manifest-testrun0000001.jsonl")
    JsonlWriter().write(records, path)
    return path


def _tar_members(tar_path: str) -> list[str]:
    with tarfile.open(tar_path) as tf:
        return [m.name for m in tf.getmembers()]


def _tar_read(tar_path: str, member: str) -> bytes:
    with tarfile.open(tar_path) as tf:
        f = tf.extractfile(member)
        assert f is not None
        return f.read()


# --- core shard content ---


def test_only_non_excluded_records_appear_in_written_tars(tmp_path: Path) -> None:
    path_a = _make_png(tmp_path, "img_a.png", "NORMAL")
    path_excl = _make_png(tmp_path, "img_excl.png", "NORMAL")
    records = [
        ManifestRecord(
            manifest_id="testrun0000001",
            path=path_a,
            filename="img_a.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=False,
        ),
        ManifestRecord(
            manifest_id="testrun0000001",
            path=path_excl,
            filename="img_excl.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=True,
            exclusion_reason="iqr:haralick_contrast",
        ),
    ]
    manifest_path = _make_manifest(tmp_path, records)
    build_shards(manifest_path, str(tmp_path / "shards"), _RATIOS)

    tar_path = str(tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000000.tar")
    members = _tar_members(tar_path)
    assert not any("img_excl" in m for m in members)
    assert any("img_a" in m for m in members)


def test_each_tar_entry_contains_image_bytes_and_label_as_utf8_text(
    tmp_path: Path,
) -> None:
    img_path = _make_png(tmp_path, "scan001.png", "ABNORMAL")
    records = [
        ManifestRecord(
            manifest_id="testrun0000001",
            path=img_path,
            filename="scan001.png",
            label="ABNORMAL",
            split="train",
            stats={},
            excluded=False,
        ),
    ]
    manifest_path = _make_manifest(tmp_path, records)
    build_shards(manifest_path, str(tmp_path / "shards"), _RATIOS)

    tar_path = str(
        tmp_path / "shards" / "train" / "ABNORMAL" / "train-abnormal-000000.tar"
    )
    cls_bytes = _tar_read(tar_path, "scan001.cls")
    assert cls_bytes.decode("utf-8") == "ABNORMAL"
    png_bytes = _tar_read(tar_path, "scan001.png")
    assert len(png_bytes) > 0


def test_tar_entries_use_image_stem_as_sample_key(tmp_path: Path) -> None:
    path_a = _make_png(tmp_path, "xray_001.png", "NORMAL")
    path_b = _make_png(tmp_path, "xray_002.png", "NORMAL")
    records = [
        ManifestRecord(
            manifest_id="testrun0000001",
            path=path_a,
            filename="xray_001.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=False,
        ),
        ManifestRecord(
            manifest_id="testrun0000001",
            path=path_b,
            filename="xray_002.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=False,
        ),
    ]
    manifest_path = _make_manifest(tmp_path, records)
    build_shards(manifest_path, str(tmp_path / "shards"), _RATIOS)

    tar_path = str(tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000000.tar")
    members = _tar_members(tar_path)
    keys = {m.rsplit(".", 1)[0] for m in members}
    assert "xray_001" in keys
    assert "xray_002" in keys


# --- batching ---


def test_five_records_with_shard_size_2_produces_3_tar_files(tmp_path: Path) -> None:
    paths = [_make_png(tmp_path, f"img_{i:03d}.png", "NORMAL") for i in range(5)]
    records = [
        ManifestRecord(
            manifest_id="testrun0000001",
            path=paths[i],
            filename=f"img_{i:03d}.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=False,
        )
        for i in range(5)
    ]
    manifest_path = _make_manifest(tmp_path, records)
    build_shards(manifest_path, str(tmp_path / "shards"), _RATIOS, shard_size=2)

    shard_dir = tmp_path / "shards" / "train" / "NORMAL"
    tar_files = list(shard_dir.glob("*.tar"))
    assert len(tar_files) == 3


# --- manifest rewrite ---


def test_manifest_is_rewritten_with_non_null_shard_field_on_non_excluded_records(
    tmp_path: Path,
) -> None:
    img_path = _make_png(tmp_path, "img001.png", "NORMAL")
    records = [
        ManifestRecord(
            manifest_id="testrun0000001",
            path=img_path,
            filename="img001.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=False,
        ),
    ]
    manifest_path = _make_manifest(tmp_path, records)
    build_shards(manifest_path, str(tmp_path / "shards"), _RATIOS)

    lines = Path(manifest_path).read_text().splitlines()
    parsed = [json.loads(ln) for ln in lines if ln.strip()]
    assert all(p["shard"] is not None for p in parsed if not p["excluded"])


def test_excluded_records_retain_null_shard_after_manifest_rewrite(
    tmp_path: Path,
) -> None:
    img_path = _make_png(tmp_path, "img001.png", "NORMAL")
    excl_path = _make_png(tmp_path, "img_excl.png", "NORMAL")
    records = [
        ManifestRecord(
            manifest_id="testrun0000001",
            path=img_path,
            filename="img001.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=False,
        ),
        ManifestRecord(
            manifest_id="testrun0000001",
            path=excl_path,
            filename="img_excl.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=True,
            exclusion_reason="lung_out_of_frame",
        ),
    ]
    manifest_path = _make_manifest(tmp_path, records)
    build_shards(manifest_path, str(tmp_path / "shards"), _RATIOS)

    parsed = [
        json.loads(ln)
        for ln in Path(manifest_path).read_text().splitlines()
        if ln.strip()
    ]
    excluded = [p for p in parsed if p["excluded"]]
    assert all(p["shard"] is None for p in excluded)


# --- shard index offset ---


def test_start_shard_index_offset_applied_to_output_tar_filename(
    tmp_path: Path,
) -> None:
    img_path = _make_png(tmp_path, "img001.png", "NORMAL")
    records = [
        ManifestRecord(
            manifest_id="testrun0000001",
            path=img_path,
            filename="img001.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=False,
        ),
    ]
    manifest_path = _make_manifest(tmp_path, records)
    build_shards(
        manifest_path,
        str(tmp_path / "shards"),
        _RATIOS,
        start_shard_index={("train", "NORMAL"): 5},
    )

    expected = tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000005.tar"
    assert expected.exists()


# --- split report ---


def test_split_report_json_is_written_with_required_top_level_keys(
    tmp_path: Path,
) -> None:
    img_path = _make_png(tmp_path, "img001.png", "NORMAL")
    records = [
        ManifestRecord(
            manifest_id="testrun0000001",
            path=img_path,
            filename="img001.png",
            label="NORMAL",
            split="train",
            stats={},
            excluded=False,
        ),
    ]
    manifest_path = _make_manifest(tmp_path, records)
    build_shards(manifest_path, str(tmp_path / "shards"), _RATIOS)

    report_path = tmp_path / "split-report-testrun0000001.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert "run_id" in report
    assert "configured_ratios" in report
    assert "observed" in report
