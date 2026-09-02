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

import multiprocessing as mp
import os
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, TypeVar

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from radiologist.etl import optional
from radiologist.etl.beam_executor import BeamExecutor
from radiologist.etl.models import BatchOutcome, ShardJob, ShardOutcome

T = TypeVar("T")
R = TypeVar("R")

BatchMapper = Callable[[Sequence[Sequence[str]]], list[BatchOutcome]]
ShardMapper = Callable[[Sequence[ShardJob]], list[ShardOutcome]]


def storage_options_from_cfg(cfg: DictConfig) -> dict | None:
    """Pull a plain ``dict`` out of ``cfg.storage_options``, or ``None``.

    An explicitly configured empty ``storage_options: {}`` is a deliberate,
    valid "no special options" value and is returned as ``{}``, not
    collapsed to ``None`` — only an absent/null key yields ``None``.

    Args:
        cfg: a Hydra ``DictConfig`` that may carry a ``storage_options`` key.

    Returns:
        A plain ``dict`` (possibly empty), or ``None`` when the key is
        absent or explicitly ``null``.
    """
    raw = (
        OmegaConf.to_container(cfg.storage_options)
        if OmegaConf.select(cfg, "storage_options") is not None
        else None
    )
    return dict(raw) if isinstance(raw, dict) else None


def default_workers() -> int:
    """The single source of truth for the default worker count.

    Returns:
        ``os.cpu_count()`` or ``1``; every caller that needs a default
        worker count defers to this function.
    """
    return os.cpu_count() or 1


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
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size!r}")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


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
    resolved_workers = workers if workers is not None else default_workers()
    resolved_max_pending = (
        max_pending if max_pending is not None else resolved_workers * 2
    )

    def _mapper(items: Sequence[T]) -> list[R]:
        items = list(items)
        results: list[Any] = [None] * len(items)
        pending: dict[Any, int] = {}
        indices = iter(enumerate(items))

        with ProcessPoolExecutor(
            max_workers=resolved_workers, mp_context=mp.get_context("spawn")
        ) as pool:

            def _submit_next() -> bool:
                try:
                    idx, item = next(indices)
                except StopIteration:
                    return False
                future = pool.submit(fn, item)
                pending[future] = idx
                return True

            for _ in range(min(resolved_max_pending, len(items))):
                _submit_next()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = pending.pop(future)
                    results[idx] = future.result()
                    _submit_next()

        return results

    return _mapper


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


_SUPPORTED_FAMILIES = ("local", "dask", "ray", "beam")

_BACKEND_EXTRAS = {
    "dask": "dask",
    "ray": "ray",
    "beam": "beam",
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Fetch ``key`` from a ``DictConfig``, plain ``dict``, or ``None``."""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(cfg, key, default)


def _backend_available(family: str) -> bool:
    if family == "local":
        return bool(optional._PREFECT_AVAILABLE)
    if family == "dask":
        return bool(optional._PREFECT_DASK_AVAILABLE)
    if family == "ray":
        return bool(optional._PREFECT_RAY_AVAILABLE)
    if family == "beam":
        return bool(optional._BEAM_AVAILABLE)
    return False


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
    if runner_cfg is None:
        return ExecutionPlan(
            family="local",
            task_runner=None,
            beam=None,
            batch_size=batch_size if batch_size is not None else 64,
        )

    family = _cfg_get(runner_cfg, "family", "local") or "local"
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unknown runner family: {family!r}. "
            f"Supported families: {', '.join(_SUPPORTED_FAMILIES)}"
        )

    resolved_batch_size = (
        batch_size if batch_size is not None else _cfg_get(runner_cfg, "batch_size", 64)
    )

    if not _backend_available(family):
        extra = _BACKEND_EXTRAS.get(family, "prefect")
        raise RuntimeError(
            f"the {family} extra is required to use the {family} runner family. "
            f"Install with: pip install 'radiologist-etl[{extra}]'"
        )

    if family == "beam":
        beam_node = _cfg_get(runner_cfg, "beam")
        beam_obj = instantiate(beam_node) if beam_node is not None else None
        return ExecutionPlan(
            family=family,
            task_runner=None,
            beam=beam_obj,
            batch_size=resolved_batch_size,
        )

    task_runner_node = _cfg_get(runner_cfg, "task_runner")
    task_runner = (
        instantiate(task_runner_node) if task_runner_node is not None else None
    )
    return ExecutionPlan(
        family=family,
        task_runner=task_runner,
        beam=None,
        batch_size=resolved_batch_size,
    )
