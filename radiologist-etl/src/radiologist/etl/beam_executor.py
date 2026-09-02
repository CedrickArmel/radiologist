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

"""Apache Beam batch executor — structural peer of Dask/Ray, implemented later.

Beam is not a ``prefect.task_runners.TaskRunner`` subclass, so it cannot be
wired the same way as the Dask/Ray runners. Instead it is one concrete class
matching the :data:`~radiologist.etl.execution.BatchMapper` callable shape,
run inside a normal Prefect task. Stub only in this issue; the real
implementation lands in the deferred Beam issue (#189).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from radiologist.etl.models import BatchOutcome, ShardJob, ShardOutcome
from radiologist.etl.stats import StatExtractor


class BeamExecutor:
    """Runs a batch mapper as an Apache Beam pipeline inside a single Prefect task."""

    def __init__(
        self,
        pipeline_options: Mapping[str, Any],
        parts_dir: str,
        storage_options: dict | None = None,
    ) -> None:
        """Configure the Beam pipeline.

        Args:
            pipeline_options: Beam ``PipelineOptions`` keyword arguments.
            parts_dir: scratch directory for intermediate Beam output.
            storage_options: extra kwargs forwarded to fsspec.
        """
        self.pipeline_options = dict(pipeline_options)
        self.parts_dir = parts_dir
        self.storage_options = storage_options

    def run_batches(
        self,
        batches: Sequence[Sequence[str]],
        images_root: str | None,
        masks_root: str | None,
        manifest_id: str,
        extractors: list[StatExtractor],
    ) -> list[BatchOutcome]:
        """Run every batch through a Beam pipeline and collect the outcomes.

        Args:
            batches: batches of image path sequences to process.
            images_root: root directory used to resolve mask mirror paths.
            masks_root: root directory of masks; None when unavailable.
            manifest_id: run identifier stamped on every produced record.
            extractors: list of StatExtractor callables.

        Returns:
            One :class:`~radiologist.etl.models.BatchOutcome` per input batch.
        """
        raise NotImplementedError

    def run_shards(self, jobs: Sequence[ShardJob]) -> list[ShardOutcome]:
        """Run every shard-writing job through a Beam pipeline and collect the outcomes.

        This is the build stage's counterpart to :meth:`run_batches` — the
        method the build flow hands to :func:`~radiologist.etl.build.build_shards`
        as its :data:`~radiologist.etl.execution.ShardMapper` when the resolved
        plan carries a Beam executor. Stub only in this issue; the real
        implementation lands in the deferred Beam issue (#189).

        Args:
            jobs: shard work units to run.

        Returns:
            One :class:`~radiologist.etl.models.ShardOutcome` per input job.
        """
        raise NotImplementedError
