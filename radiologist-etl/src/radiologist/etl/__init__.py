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

from radiologist.etl.filters import filter_iqr, filter_lung_out_of_frame
from radiologist.etl.manifest import (
    JsonlWriter,
    ManifestRecord,
    ParquetWriter,
    records_reader,
)
from radiologist.etl.ops import compute_run_id
from radiologist.etl.prefect_pipelines import (
    apply_filters_task,
    assign_splits_task,
    build_shards_task,
    compute_stats_task,
    etl_flow,
    write_jsonl_task,
)
from radiologist.etl.processors import StatsProcessor, lung_out_of_frame
from radiologist.etl.shards import build_shards
from radiologist.etl.split import assign_split
from radiologist.etl.stats import StatExtractor, lung_asymmetry, make_haralick

__all__: list[str] = [
    "apply_filters_task",
    "assign_splits_task",
    "assign_split",
    "build_shards",
    "build_shards_task",
    "compute_run_id",
    "compute_stats_task",
    "etl_flow",
    "filter_iqr",
    "filter_lung_out_of_frame",
    "JsonlWriter",
    "lung_asymmetry",
    "lung_out_of_frame",
    "make_haralick",
    "ManifestRecord",
    "ParquetWriter",
    "records_reader",
    "StatExtractor",
    "StatsProcessor",
    "write_jsonl_task",
]
