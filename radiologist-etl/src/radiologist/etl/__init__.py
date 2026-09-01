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

"""ETL pipeline: outlier filtering, splitting, manifests, and shard building."""

from __future__ import annotations

from radiologist.etl.assign import assign_splits
from radiologist.etl.beam_executor import BeamExecutor
from radiologist.etl.build import build_shards
from radiologist.etl.execution import ExecutionPlan, default_workers, resolve_execution
from radiologist.etl.extract import ExtractionFailureError, extract
from radiologist.etl.filters import filter_iqr, filter_lung_out_of_frame
from radiologist.etl.identity import (
    compute_assign_run_id,
    compute_build_run_id,
    compute_extract_run_id,
    config_digest,
    content_digest,
    directory_digest,
)
from radiologist.etl.manifest import (
    JsonlWriter,
    ManifestRecord,
    ParquetWriter,
    records_reader,
)
from radiologist.etl.models import (
    AssignSplitResult,
    BatchOutcome,
    BuildResult,
    EtlResult,
    ExtractResult,
    ShardJob,
    ShardOutcome,
)
from radiologist.etl.ops import compute_run_id
from radiologist.etl.prefect_pipelines import (
    apply_filters_task,
    assign_split_flow,
    assign_splits_task,
    build_flow,
    build_shards_task,
    compute_stats_task,
    etl_flow,
    extract_flow,
    run_assign_split,
    run_build,
    run_extract,
    write_jsonl_task,
)
from radiologist.etl.processors import StatsProcessor, lung_out_of_frame, process_batch
from radiologist.etl.shards import plan_shards, write_shard
from radiologist.etl.split import SplitRatios, assign_split, normalize_ratios
from radiologist.etl.stats import StatExtractor, lung_asymmetry, make_haralick

__all__: list[str] = [
    "apply_filters_task",
    "assign_split",
    "assign_split_flow",
    "AssignSplitResult",
    "assign_splits",
    "assign_splits_task",
    "BatchOutcome",
    "BeamExecutor",
    "build_flow",
    "build_shards",
    "build_shards_task",
    "BuildResult",
    "compute_assign_run_id",
    "compute_build_run_id",
    "compute_extract_run_id",
    "compute_run_id",
    "compute_stats_task",
    "config_digest",
    "content_digest",
    "default_workers",
    "directory_digest",
    "etl_flow",
    "EtlResult",
    "ExecutionPlan",
    "extract",
    "extract_flow",
    "ExtractionFailureError",
    "ExtractResult",
    "filter_iqr",
    "filter_lung_out_of_frame",
    "JsonlWriter",
    "lung_asymmetry",
    "lung_out_of_frame",
    "make_haralick",
    "ManifestRecord",
    "normalize_ratios",
    "ParquetWriter",
    "plan_shards",
    "process_batch",
    "records_reader",
    "resolve_execution",
    "run_assign_split",
    "run_build",
    "run_extract",
    "ShardJob",
    "ShardOutcome",
    "SplitRatios",
    "StatExtractor",
    "StatsProcessor",
    "write_jsonl_task",
    "write_shard",
]
