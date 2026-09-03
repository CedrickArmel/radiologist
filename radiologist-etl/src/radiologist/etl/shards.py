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

"""Build WebDataset tar shards from a split manifest."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import fsspec  # type: ignore[import-untyped]
import webdataset as wds  # type: ignore[import-untyped]

import radiologist.utils.filesystem as fst
from radiologist.etl.manifest import ManifestRecord
from radiologist.etl.models import ShardJob, ShardOutcome


def plan_shards(
    records: Sequence[ManifestRecord],
    shard_root: str,
    shard_size: int = 1000,
) -> list[ShardJob]:
    """Group non-excluded records into deterministic per-shard work units.

    Args:
        records: manifest records to plan shards for.
        shard_root: directory the resulting shards will be written under.
        shard_size: max samples per shard.

    Returns:
        Records grouped by ``(split, label)`` in ascending key order, chunked
        into ``shard_size`` units, indexed from 0 within each group;
        deterministic for a given record order.

    Raises:
        ValueError: if ``shard_size < 1``.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size!r}")

    groups: dict[tuple[str, str], list[ManifestRecord]] = defaultdict(list)
    for record in records:
        if not record.excluded:
            groups[(record.split, record.label)].append(record)

    jobs: list[ShardJob] = []
    for key in sorted(groups.keys()):
        split, label = key
        group_records = groups[key]
        for idx, chunk_start in enumerate(range(0, len(group_records), shard_size)):
            chunk = group_records[chunk_start : chunk_start + shard_size]
            jobs.append(
                ShardJob(
                    split=split,
                    label=label,
                    index=idx,
                    shard_root=shard_root,
                    records=chunk,
                )
            )
    return jobs


def write_shard(
    job: ShardJob,
    storage_options: dict | None = None,
) -> ShardOutcome:
    """Write one WebDataset tar shard for a planned job.

    Top-level (picklable) function, safe to cross a process boundary.

    Args:
        job: the shard's work unit, as planned by :func:`plan_shards`.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        A :class:`~radiologist.etl.models.ShardOutcome` describing the write.
        Per-image read errors are collected into ``failures`` rather than raised.
    """
    opts = storage_options or {}
    shard_filename = f"{job.split}-{job.label.lower()}-{job.index:06d}.tar"
    parts = [p for p in (job.split, job.label, shard_filename) if p]
    relative_path = "/".join(parts)
    shard_path = fst.pathjoin(job.shard_root, *parts)

    fs_dst, dst_path = fsspec.url_to_fs(shard_path, **opts)
    if hasattr(fs_dst, "makedirs"):
        fs_dst.makedirs(fst.pathparent(dst_path), exist_ok=True)

    record_paths: list[str] = []
    failures: list[tuple[str, str]] = []
    written = 0

    with fs_dst.open(dst_path, "wb") as out:
        with wds.TarWriter(out) as sink:
            for record in job.records:
                try:
                    fs_src, src_path = fsspec.url_to_fs(record.path, **opts)
                    with fs_src.open(src_path, "rb") as img_f:
                        img_bytes = img_f.read()
                except OSError as exc:
                    failures.append((record.path, str(exc)))
                    continue

                stem = fst.pathstem(record.filename)
                sink.write(
                    {
                        "__key__": stem,
                        "png": img_bytes,
                        "cls": record.label.encode(),
                    }
                )
                record_paths.append(record.path)
                written += 1

    return ShardOutcome(
        relative_path=relative_path,
        record_paths=record_paths,
        written=written,
        failures=failures,
    )
