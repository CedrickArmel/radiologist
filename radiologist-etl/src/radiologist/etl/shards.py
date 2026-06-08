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
import math
from collections import defaultdict

import fsspec  # type: ignore[import-untyped]
import webdataset as wds  # type: ignore[import-untyped]
from rich.progress import Progress

import radiologist.utils.filesystem as fst
from radiologist.etl.manifest import JsonlWriter, ManifestRecord, records_reader


def build_shards(
    manifest_path: str,
    shard_root: str,
    ratios: dict[str, float],
    shard_size: int = 1000,
    start_shard_index: dict[tuple[str, str], int] | None = None,
    storage_options: dict | None = None,
) -> str:
    """Build WebDataset tar shards from a JSONL manifest.

    Args:
        manifest_path: path or URI to the manifest-{run_id}.jsonl file.
        shard_root: directory where shards are written.
        ratios: configured split ratios (for the split report).
        shard_size: max samples per shard.
        start_shard_index: per-(split, label) shard index offset for incremental runs.
        storage_options: extra kwargs for fsspec.

    Returns:
        Updated manifest_path (same path, rewritten in place).
    """
    opts = storage_options or {}
    if start_shard_index is None:
        start_shard_index = {}

    records = records_reader(manifest_path, opts)

    groups: dict[tuple[str, str], list[ManifestRecord]] = defaultdict(list)
    for rec in records:
        if not rec.excluded:
            groups[(rec.split, rec.label)].append(rec)

    total_shards = sum(math.ceil(len(g) / shard_size) for g in groups.values())
    with Progress() as progress:
        task_id = progress.add_task("Building shards...", total=total_shards)
        for key, group_records in groups.items():
            split, label = key
            idx = start_shard_index.get(key, 0)

            for chunk_start in range(0, len(group_records), shard_size):
                chunk = group_records[chunk_start : chunk_start + shard_size]
                shard_filename = f"{split}-{label.lower()}-{idx:06d}.tar"
                shard_path = fst.pathjoin(shard_root, split, label, shard_filename)
                fs_dst, dst_path = fsspec.url_to_fs(shard_path, **opts)
                relative_shard = fs_dst.sep.join([split, label, shard_filename])
                if hasattr(fs_dst, "makedirs"):
                    fs_dst.makedirs(fst.pathparent(dst_path), exist_ok=True)
                with fs_dst.open(dst_path, "wb") as out:
                    with wds.TarWriter(out) as sink:
                        for record in chunk:
                            fs_src, src_path = fsspec.url_to_fs(record.path, **opts)
                            with fs_src.open(src_path, "rb") as img_f:
                                img_bytes = img_f.read()
                            stem = fst.pathstem(record.filename)
                            sink.write(
                                {
                                    "__key__": stem,
                                    "png": img_bytes,
                                    "cls": record.label.encode(),
                                }
                            )
                            record.shard = relative_shard
                progress.advance(task_id)
                idx += 1

    JsonlWriter().write(records, manifest_path, storage_options=storage_options)

    manifest_stem = fst.pathstem(manifest_path)
    run_id = manifest_stem.split("-", 1)[1] if "-" in manifest_stem else manifest_stem
    manifest_parent = fst.pathparent(manifest_path)
    report_path = fst.pathjoin(manifest_parent, f"split-report-{run_id}.json")

    label_split_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for rec in records:
        if rec.excluded:
            label_split_counts[rec.label]["excluded"] += 1
        else:
            label_split_counts[rec.label][rec.split] += 1

    observed: dict[str, dict[str, float]] = {}
    for label, split_counts in label_split_counts.items():
        total_all = sum(split_counts.values())
        observed[label] = {
            split: (count / total_all) for split, count in split_counts.items()
        }

    report = {
        "run_id": run_id,
        "configured_ratios": ratios,
        "observed": observed,
    }
    fs_r, rpath = fsspec.url_to_fs(report_path, **opts)
    if hasattr(fs_r, "makedirs"):
        fs_r.makedirs(fst.pathparent(rpath), exist_ok=True)
    with fs_r.open(rpath, "wt", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    return manifest_path
