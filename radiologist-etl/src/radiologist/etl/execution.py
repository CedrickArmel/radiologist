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

"""Runner-selection wiring: local mapper, batch chunking, and ExecutionPlan.

Multi-backend execution rides Prefect's own ``TaskRunner`` abstraction plus
Hydra ``_target_`` instantiation — there is no custom ``ExecutionBackend``/
``Runner`` Protocol. Parallelism enters the pure stage functions through a
single injected callable (:data:`BatchMapper` for extract, :data:`ShardMapper`
for build); the default is a bounded in-process pool, so the stage functions
stay plain Python and fully testable without Prefect. :class:`ExecutionPlan`
is the only shared type: it tells a flow which wiring path to take.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from radiologist.etl.beam_executor import BeamExecutor
from radiologist.etl.models import BatchOutcome, ShardJob, ShardOutcome

T = TypeVar("T")
R = TypeVar("R")

BatchMapper = Callable[[Sequence[Sequence[str]]], list[BatchOutcome]]
ShardMapper = Callable[[Sequence[ShardJob]], list[ShardOutcome]]


def default_workers() -> int:
    """The single source of truth for the default worker count.

    Returns:
        ``os.cpu_count()`` or ``1``; every caller that needs a default
        worker count defers to this function.
    """
    raise NotImplementedError


def chunked(items: Sequence[T], size: int) -> list[list[T]]:
    """Split a sequence into consecutive chunks of at most ``size`` items.

    Args:
        items: the sequence to chunk.
        size: maximum chunk size; must be >= 1.

    Returns:
        Consecutive chunks; the last chunk may be shorter; ``[]`` for empty input.

    Raises:
        ValueError: if ``size < 1``.
    """
    raise NotImplementedError


def local_mapper(
    fn: Callable[[T], R],
    workers: int | None = None,
    max_pending: int | None = None,
) -> Callable[[Sequence[T]], list[R]]:
    """Build a bounded, in-process-pool mapper over ``fn``.

    Args:
        fn: the picklable callable to apply to each item.
        workers: pool size; defaults to :func:`default_workers`.
        max_pending: max outstanding units at a time; defaults to ``workers * 2``.

    Returns:
        A callable that applies ``fn`` across the pool and returns results in
        input order.
    """
    raise NotImplementedError


@dataclass(frozen=True)
class ExecutionPlan:
    """Which of the wiring paths a stage flow should take.

    Attributes:
        family: one of ``"local"``, ``"dask"``, ``"ray"``, ``"beam"``.
        task_runner: a ``prefect.task_runners.TaskRunner`` instance, or None.
        beam: a configured :class:`~radiologist.etl.beam_executor.BeamExecutor`, or None.
        batch_size: number of image paths per dispatched batch/job.
    """

    family: str
    task_runner: Any | None = None
    beam: BeamExecutor | None = None
    batch_size: int = 64


def resolve_execution(
    runner_cfg: Any | None = None,
    batch_size: int | None = None,
) -> ExecutionPlan:
    """Resolve a Hydra ``runner`` config node into an :class:`ExecutionPlan`.

    Args:
        runner_cfg: the ``runner`` config node (or None/absent for local).
        batch_size: number of image paths per dispatched batch/job.

    Returns:
        For local/dask/ray, instantiates ``runner_cfg.task_runner`` via
        ``hydra.utils.instantiate`` and returns it in ``ExecutionPlan.task_runner``;
        for beam, instantiates ``runner_cfg.beam`` into ``ExecutionPlan.beam``;
        a None/absent node yields ``family="local"`` with no task runner.

    Raises:
        ValueError: on an unknown family.
        RuntimeError: naming the extra to install when the family's backend
            package is unavailable.
    """
    raise NotImplementedError
