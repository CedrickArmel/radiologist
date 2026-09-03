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

"""Build stage: one split manifest → WebDataset tar shards + manifest + report.

Writes shards, a shard-annotated manifest, and a split report under
``{shard_root}/{run_id}``. Never mutates the input split manifest. The
``ratios`` argument is used only for the report's configured-vs-observed
comparison — never for assignment (assignment already happened in the
assign-split stage).

This module's :func:`build_shards` is the package-level public
``radiologist.etl.build_shards`` (rebound here by issue #185). The old,
differently-signatured ``radiologist.etl.shards.build_shards`` (in-place
manifest rewrite, dict ratios) predated the three-stage redesign and has
been removed.
"""

from __future__ import annotations

import functools
import json
from collections import defaultdict

import fsspec  # type: ignore[import-untyped]

import radiologist.utils.filesystem as fst
from radiologist.etl.execution import ShardMapper, default_workers, local_mapper
from radiologist.etl.identity import compute_build_run_id
from radiologist.etl.manifest import JsonlWriter, records_reader
from radiologist.etl.models import BuildResult
from radiologist.etl.shards import plan_shards, write_shard
from radiologist.etl.split import SplitRatios


class BuildFailureError(RuntimeError):
    """Raised when the share of unsharded records exceeds max_failure_rate."""


# Exclusion reason code stamped on a record whose image could not be written
# into its tar shard. Reason codes are pipe-joined when a record accumulates
# more than one.
SHARD_WRITE_FAILED_REASON: str = "shard_write_failed"


def build_shards(
    split_manifest_path: str,
    shard_root: str,
    shard_size: int = 1000,
    ratios: SplitRatios | None = None,
    workers: int | None = None,
    run_label: str | None = None,
    mapper: ShardMapper | None = None,
    storage_options: dict | None = None,
    max_failure_rate: float = 0.0,
) -> BuildResult:
    """Build WebDataset tar shards, a shard-annotated manifest, and a split report.

    Args:
        split_manifest_path: path to the split manifest produced by the
            assign-split stage.
        shard_root: directory shards are written under (inside
            ``{shard_root}/{run_id}``).
        shard_size: max samples per shard.
        ratios: configured split ratios, used only for the report.
        workers: worker count; defaults to :func:`~radiologist.etl.execution.default_workers`.
        run_label: optional label folded into the run id.
        mapper: shard-dispatching callable; defaults to a local process-pool mapper.
        storage_options: extra kwargs forwarded to fsspec.
        max_failure_rate: tolerated share of records that cannot be written into
            a shard. Accepted but not yet enforced.

    Returns:
        A :class:`~radiologist.etl.models.BuildResult` describing the run.

    Raises:
        FileNotFoundError: if ``split_manifest_path`` does not exist.
        ValueError: if ``shard_size < 1``.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size!r}")

    opts = storage_options or {}
    fs, fs_path = fsspec.url_to_fs(split_manifest_path, **opts)
    if not fs.exists(fs_path):
        raise FileNotFoundError(f"Split manifest not found: {split_manifest_path}")

    config: dict = {"shard_size": shard_size}
    if ratios is not None:
        config["ratios"] = [list(pair) for pair in ratios]
    if run_label is not None:
        config["run_label"] = run_label
    run_id = compute_build_run_id(split_manifest_path, config, storage_options)

    output_dir = fst.pathjoin(shard_root, run_id)

    records = records_reader(split_manifest_path, opts)

    jobs = plan_shards(records, output_dir, shard_size)

    if mapper is None:
        resolved_workers = workers if workers is not None else default_workers()
        mapper = local_mapper(
            functools.partial(write_shard, storage_options=storage_options),
            workers=resolved_workers,
        )

    outcomes = mapper(jobs) if jobs else []

    path_to_shard: dict[str, str] = {}
    for outcome in outcomes:
        for record_path in outcome.record_paths:
            path_to_shard[record_path] = outcome.relative_path

    for record in records:
        if not record.excluded:
            record.shard = path_to_shard.get(record.path)

    manifest_path = fst.pathjoin(output_dir, f"manifest-{run_id}.jsonl")
    JsonlWriter().write(records, manifest_path, storage_options=storage_options)

    label_split_counts: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for record in records:
        if record.excluded:
            label_split_counts[record.label]["excluded"] += 1
        else:
            label_split_counts[record.label][record.split] += 1

    observed: dict[str, dict[str, float]] = {}
    for label, split_counts in label_split_counts.items():
        total_all = sum(split_counts.values())
        observed[label] = {
            split: (count / total_all) for split, count in split_counts.items()
        }

    report = {
        "run_id": run_id,
        "configured_ratios": list(ratios) if ratios is not None else [],
        "observed": observed,
    }
    report_path = fst.pathjoin(output_dir, f"split-report-{run_id}.json")
    fs_r, rpath = fsspec.url_to_fs(report_path, **opts)
    if hasattr(fs_r, "makedirs"):
        fs_r.makedirs(fst.pathparent(rpath), exist_ok=True)
    with fs_r.open(rpath, "wt", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    shard_count = len(outcomes)
    record_count = sum(outcome.written for outcome in outcomes)

    return BuildResult(
        run_id=run_id,
        output_dir=output_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        shard_count=shard_count,
        record_count=record_count,
    )
