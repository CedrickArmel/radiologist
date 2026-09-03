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

"""Lightweight result dataclasses returned by ETL entry points."""

from __future__ import annotations

from dataclasses import dataclass

from radiologist.etl.manifest import ManifestRecord


@dataclass(frozen=True)
class ExtractResult:
    """Outcome of one extract-stage run.

    Attributes:
        run_id: 16-char content-addressed id for this extract run.
        manifest_path: ``{destination}/extract-{run_id}.jsonl``.
        total: images listed in the input file listing.
        succeeded: images that produced a record.
        failed: images that raised and were recorded as failures.
        failure_rate: ``failed / total``, ``0.0`` when ``total == 0``.
        excluded: records flagged excluded by the quality filters.
    """

    run_id: str
    manifest_path: str
    total: int
    succeeded: int
    failed: int
    failure_rate: float
    excluded: int


@dataclass(frozen=True)
class AssignSplitResult:
    """Outcome of one assign-split-stage run.

    Attributes:
        run_id: 16-char content-addressed id for this assign-split run.
        split_manifest_path: ``{destination}/manifest-{run_id}.jsonl``.
        source_manifest_count: extract manifests read from the input folder.
        record_count: records written after deduplication.
        duplicate_count: records dropped as duplicates of an earlier record.
        counts_by_split: number of records assigned to each split name.
    """

    run_id: str
    split_manifest_path: str
    source_manifest_count: int
    record_count: int
    duplicate_count: int
    counts_by_split: dict[str, int]


@dataclass(frozen=True)
class BuildResult:
    """Outcome of one build-stage run.

    Attributes:
        run_id: 16-char content-addressed id for this build run.
        output_dir: ``{shard_root}/{run_id}``.
        manifest_path: ``{output_dir}/manifest-{run_id}.jsonl``, shard field populated.
        report_path: ``{output_dir}/split-report-{run_id}.json``.
        shard_count: number of tar shards written.
        record_count: non-excluded records written into shards.
        failed: records that were planned into a shard but could not be written.
        failure_rate: ``failed / planned``, ``0.0`` when nothing was planned.
    """

    run_id: str
    output_dir: str
    manifest_path: str
    report_path: str
    shard_count: int
    record_count: int
    failed: int = 0
    failure_rate: float = 0.0


@dataclass(frozen=True)
class BatchOutcome:
    """Result of processing one batch of image paths.

    Attributes:
        records: one :class:`ManifestRecord` per readable image.
        failures: ``(image path, error message)`` per unreadable image.
    """

    records: list[ManifestRecord]
    failures: list[tuple[str, str]]


@dataclass(frozen=True)
class ShardJob:
    """One tar shard's worth of work; self-contained so it can cross a process boundary.

    Attributes:
        split: dataset split name (train/val/test).
        label: class label.
        index: shard index within the (split, label) group.
        shard_root: directory where shards are written.
        records: records assigned to this shard.
    """

    split: str
    label: str
    index: int
    shard_root: str
    records: list[ManifestRecord]


@dataclass(frozen=True)
class ShardOutcome:
    """Result of writing one tar shard.

    Attributes:
        relative_path: ``{split}/{label}/{split}-{label}-{index:06d}.tar``.
        record_paths: source paths written into this shard, in order.
        written: number of images successfully written.
        failures: ``(image path, error message)`` per unreadable image.
    """

    relative_path: str
    record_paths: list[str]
    written: int
    failures: list[tuple[str, str]]
