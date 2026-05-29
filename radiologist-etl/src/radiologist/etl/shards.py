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
from collections import defaultdict
from pathlib import Path

import fsspec  # type: ignore[import-untyped]
import webdataset as wds  # type: ignore[import-untyped]

from radiologist.etl.manifest import JsonlWriter, ManifestRecord


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
    if start_shard_index is None:
        start_shard_index = {}

    # Read all records from manifest
    fs_m, mpath = fsspec.url_to_fs(manifest_path, **(storage_options or {}))
    records: list[ManifestRecord] = []
    with fs_m.open(mpath, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ManifestRecord.from_flat_dict(json.loads(line)))

    # Group non-excluded records by (split, label)
    groups: dict[tuple[str, str], list[ManifestRecord]] = defaultdict(list)
    for rec in records:
        if not rec.excluded:
            groups[(rec.split, rec.label)].append(rec)

    for key, group_records in groups.items():
        split, label = key
        idx = start_shard_index.get(key, 0)

        for chunk_start in range(0, max(len(group_records), 1), shard_size):
            chunk = group_records[chunk_start : chunk_start + shard_size]
            shard_filename = f"{split}-{label.lower()}-{idx:06d}.tar"
            shard_path = f"{shard_root}/{split}/{label}/{shard_filename}"
            relative_shard = f"{split}/{label}/{shard_filename}"
            Path(shard_path).parent.mkdir(parents=True, exist_ok=True)
            with wds.TarWriter(shard_path) as sink:
                for record in chunk:
                    fs_src, src_path = fsspec.url_to_fs(
                        record.path, **(storage_options or {})
                    )
                    with fs_src.open(src_path, "rb") as img_f:
                        img_bytes = img_f.read()
                    stem = Path(record.filename).stem
                    sink.write(
                        {
                            "__key__": stem,
                            "png": img_bytes,
                            "cls": record.label.encode(),
                        }
                    )
                    record.shard = relative_shard
            idx += 1

    # Write updated manifest in place
    JsonlWriter().write(records, manifest_path, storage_options=storage_options)

    # Compute split report
    run_id = Path(manifest_path).stem.split("-", 1)[1]
    report_path = str(Path(manifest_path).parent / f"split-report-{run_id}.json")

    # Count per label, per split (including excluded as its own bucket)
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
            split: count / total_all for split, count in split_counts.items()
        }

    report = {
        "run_id": run_id,
        "configured_ratios": ratios,
        "observed": observed,
    }
    fs_r, rpath = fsspec.url_to_fs(report_path, **(storage_options or {}))
    with fs_r.open(rpath, "wt", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    return manifest_path
