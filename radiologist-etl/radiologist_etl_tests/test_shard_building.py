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

import tarfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from radiologist.etl import ManifestRecord


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(str(path))


def _make_png(root: Path, filename: str, label: str) -> str:
    dest = root / label / filename
    _write_png(dest, np.zeros((10, 10, 3), dtype=np.uint8))
    return str(dest)


def _tar_members(tar_path: str) -> list[str]:
    with tarfile.open(tar_path) as tf:
        return [m.name for m in tf.getmembers()]


# --- plan_shards --------------------------------------------------------------


def _record(path: str, filename: str, label: str, split: str, excluded=False):
    return ManifestRecord(
        manifest_id="testrun0000001",
        path=path,
        filename=filename,
        label=label,
        split=split,
        stats={},
        excluded=excluded,
    )


def test_plan_shards_raises_value_error_for_shard_size_below_one():
    from radiologist.etl.shards import plan_shards

    with pytest.raises(ValueError):
        plan_shards([], "/root", shard_size=0)


def test_plan_shards_groups_records_by_split_and_label_and_indexes_from_zero():
    from radiologist.etl.shards import plan_shards

    records = [
        _record("/a.png", "a.png", "NORMAL", "train"),
        _record("/b.png", "b.png", "ABNORMAL", "train"),
    ]
    jobs = plan_shards(records, "/root", shard_size=10)

    assert len(jobs) == 2
    keys = {(j.split, j.label, j.index) for j in jobs}
    assert keys == {("train", "ABNORMAL", 0), ("train", "NORMAL", 0)}


def test_plan_shards_chunks_a_group_larger_than_shard_size_into_several_shards():
    from radiologist.etl.shards import plan_shards

    records = [_record(f"/{i}.png", f"{i}.png", "NORMAL", "train") for i in range(5)]
    jobs = plan_shards(records, "/root", shard_size=2)

    normal_jobs = sorted(
        (j for j in jobs if j.split == "train" and j.label == "NORMAL"),
        key=lambda j: j.index,
    )
    assert [j.index for j in normal_jobs] == [0, 1, 2]
    assert [len(j.records) for j in normal_jobs] == [2, 2, 1]


def test_plan_shards_numbers_each_group_independently_from_zero():
    from radiologist.etl.shards import plan_shards

    records = [_record(f"/n{i}.png", f"n{i}.png", "NORMAL", "train") for i in range(3)]
    records += [
        _record(f"/a{i}.png", f"a{i}.png", "ABNORMAL", "train") for i in range(3)
    ]
    jobs = plan_shards(records, "/root", shard_size=2)

    normal_indices = sorted(j.index for j in jobs if j.label == "NORMAL")
    abnormal_indices = sorted(j.index for j in jobs if j.label == "ABNORMAL")
    assert normal_indices == [0, 1]
    assert abnormal_indices == [0, 1]


def test_plan_shards_excludes_excluded_records():
    from radiologist.etl.shards import plan_shards

    records = [
        _record("/a.png", "a.png", "NORMAL", "train"),
        _record("/b.png", "b.png", "NORMAL", "train", excluded=True),
    ]
    jobs = plan_shards(records, "/root", shard_size=10)

    assert len(jobs) == 1
    assert len(jobs[0].records) == 1
    assert jobs[0].records[0].path == "/a.png"


def test_plan_shards_returns_no_jobs_for_no_records():
    from radiologist.etl.shards import plan_shards

    assert plan_shards([], "/root", shard_size=10) == []


# --- write_shard ----------------------------------------------------------------


def test_write_shard_writes_one_entry_per_record_keyed_by_stem(tmp_path: Path) -> None:
    from radiologist.etl.models import ShardJob
    from radiologist.etl.shards import write_shard

    path_a = _make_png(tmp_path, "scan001.png", "NORMAL")
    job = ShardJob(
        split="train",
        label="NORMAL",
        index=0,
        shard_root=str(tmp_path / "shards"),
        records=[_record(path_a, "scan001.png", "NORMAL", "train")],
    )
    outcome = write_shard(job)

    tar_path = str(tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000000.tar")
    members = _tar_members(tar_path)
    assert any(m.startswith("scan001.") for m in members)
    assert outcome.written == 1
    assert outcome.failures == []


def test_write_shard_reports_failure_for_unreadable_image(tmp_path: Path) -> None:
    from radiologist.etl.models import ShardJob
    from radiologist.etl.shards import write_shard

    missing_path = str(tmp_path / "missing.png")
    job = ShardJob(
        split="train",
        label="NORMAL",
        index=0,
        shard_root=str(tmp_path / "shards"),
        records=[_record(missing_path, "missing.png", "NORMAL", "train")],
    )
    outcome = write_shard(job)

    assert outcome.written == 0
    assert len(outcome.failures) == 1
    assert outcome.failures[0][0] == missing_path

    tar_path = str(tmp_path / "shards" / "train" / "NORMAL" / "train-normal-000000.tar")
    members = _tar_members(tar_path)
    assert members == []


def test_write_shard_relative_path_resolves_against_shard_root(tmp_path: Path) -> None:
    from radiologist.etl.models import ShardJob
    from radiologist.etl.shards import write_shard

    path_a = _make_png(tmp_path, "scan001.png", "NORMAL")
    shard_root = str(tmp_path / "shards")
    job = ShardJob(
        split="train",
        label="NORMAL",
        index=0,
        shard_root=shard_root,
        records=[_record(path_a, "scan001.png", "NORMAL", "train")],
    )
    outcome = write_shard(job)

    resolved = str(Path(shard_root) / outcome.relative_path)
    assert Path(resolved).exists()
