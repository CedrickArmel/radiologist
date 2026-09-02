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

"""Apache Beam batch executor — structural peer of Dask/Ray, one opaque unit.

Beam is not a ``prefect.task_runners.TaskRunner`` subclass: a Beam pipeline
owns its own parallelism and its own runner, and is invoked from *inside* a
single orchestration task rather than by distributing that task's work. So
this family is one concrete class exposing two methods whose signatures are
exactly the :data:`~radiologist.etl.execution.BatchMapper` and
:data:`~radiologist.etl.execution.ShardMapper` callable shapes the stages
already accept — nothing else in the package knows Beam exists.

A Beam pipeline cannot return a collection to its driver, so each element
writes its outcome as a JSON line under a per-run prefix beneath
``parts_dir`` and the driver reads those parts back in input order. That
keeps the driver-side contract identical to every other family. The
per-batch/per-shard functions the pipeline applies are the same top-level
:func:`~radiologist.etl.processors.process_batch` /
:func:`~radiologist.etl.shards.write_shard` the other families use; they
cross Beam's serialization boundary the same way they cross the
process-pool one.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import fsspec  # type: ignore[import-untyped]
from omegaconf import OmegaConf

import radiologist.utils.filesystem as fst
from radiologist.etl import optional
from radiologist.etl.manifest import ManifestRecord
from radiologist.etl.models import BatchOutcome, ShardJob, ShardOutcome
from radiologist.etl.processors import process_batch
from radiologist.etl.shards import write_shard
from radiologist.etl.stats import StatExtractor

# fsspec reports a bare filesystem path as ``None``; ``file``/``local`` are
# the explicit spellings of the same thing.
_LOCAL_PROTOCOLS = frozenset({None, "file", "local"})


def _plain(value: Any) -> Any:
    """Strip OmegaConf containers so Beam only ever sees plain Python values."""
    return (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )


def _is_local_uri(uri: str) -> bool:
    """Whether ``uri`` names a location only the driver's own machine can reach."""
    protocol, _ = fsspec.core.split_protocol(uri)
    return protocol in _LOCAL_PROTOCOLS


def _is_direct_runner(runner: Any) -> bool:
    """Whether ``runner`` names one of Beam's in-process direct runners.

    Matched on the name because Beam's runner selection is a string in
    ``pipeline_options``; an absent runner is Beam's own default, which is
    the direct runner.
    """
    return runner is None or "direct" in str(runner).lower()


def _final_state(result: Any) -> str:
    """The pipeline result's terminal state, falling back to ``FAILED``.

    Reading ``state`` off a result whose runner already errored can itself
    raise, and the caller is on an error path where the state is the only
    thing worth reporting — so never let this be the exception that escapes.
    """
    try:
        return str(result.state)
    except Exception:  # noqa: BLE001 - the state is best-effort diagnostics
        return str(optional.PipelineState.FAILED)


def _part_path(parts_prefix: str, index: int) -> str:
    """The JSON-lines part file one pipeline element writes its outcome to."""
    return f"{parts_prefix.rstrip('/')}/part-{index:06d}.jsonl"


def _write_part(
    parts_prefix: str,
    index: int,
    payload: dict,
    storage_options: dict | None,
) -> None:
    """Write one element's outcome as a JSON line, from inside a Beam worker."""
    fs, path = fsspec.url_to_fs(
        _part_path(parts_prefix, index), **(storage_options or {})
    )
    if hasattr(fs, "makedirs"):
        fs.makedirs(fst.pathparent(path), exist_ok=True)
    with fs.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _read_parts(
    parts_prefix: str,
    count: int,
    storage_options: dict | None,
) -> list[dict]:
    """Read back exactly the ``count`` parts this run wrote, in input order.

    Only this run's own prefix is read, so parts left behind by any earlier
    run are structurally incapable of leaking into this one's results.
    """
    payloads: list[dict] = []
    for index in range(count):
        fs, path = fsspec.url_to_fs(
            _part_path(parts_prefix, index), **(storage_options or {})
        )
        if not fs.exists(path):
            raise RuntimeError(
                f"Beam pipeline reported success but wrote no part for unit {index}: "
                f"expected {_part_path(parts_prefix, index)!r}"
            )
        with fs.open(path, "rt", encoding="utf-8") as f:
            payloads.append(json.loads(f.read()))
    return payloads


def _batch_outcome_to_json(outcome: BatchOutcome) -> dict:
    return {
        "records": [record._to_flat_dict() for record in outcome.records],
        "failures": [list(failure) for failure in outcome.failures],
    }


def _batch_outcome_from_json(payload: dict) -> BatchOutcome:
    return BatchOutcome(
        records=[ManifestRecord.from_flat_dict(d) for d in payload["records"]],
        failures=[(path, message) for path, message in payload["failures"]],
    )


def _shard_outcome_to_json(outcome: ShardOutcome) -> dict:
    return {
        "relative_path": outcome.relative_path,
        "record_paths": list(outcome.record_paths),
        "written": outcome.written,
        "failures": [list(failure) for failure in outcome.failures],
    }


def _shard_outcome_from_json(payload: dict) -> ShardOutcome:
    return ShardOutcome(
        relative_path=payload["relative_path"],
        record_paths=list(payload["record_paths"]),
        written=int(payload["written"]),
        failures=[(path, message) for path, message in payload["failures"]],
    )


def _beam_process_batch(
    element: tuple,
    images_root: str | None,
    masks_root: str | None,
    manifest_id: str,
    extractors: list[StatExtractor],
    parts_prefix: str,
    storage_options: dict | None,
) -> int:
    """Pipeline body for the extract stage: one ``(index, paths)`` element.

    Top-level function so it survives Beam's serialization boundary, and the
    very same :func:`~radiologist.etl.processors.process_batch` every other
    execution family applies.
    """
    index, paths = element
    outcome = process_batch(
        paths,
        images_root=images_root,
        masks_root=masks_root,
        manifest_id=manifest_id,
        extractors=extractors,
        storage_options=storage_options,
    )
    _write_part(parts_prefix, index, _batch_outcome_to_json(outcome), storage_options)
    return index


def _beam_write_shard(
    element: tuple,
    parts_prefix: str,
    storage_options: dict | None,
) -> int:
    """Pipeline body for the build stage: one ``(index, job)`` element."""
    index, job = element
    outcome = write_shard(job, storage_options=storage_options)
    _write_part(parts_prefix, index, _shard_outcome_to_json(outcome), storage_options)
    return index


class BeamExecutor:
    """Runs batches of ETL work as an Apache Beam pipeline inside one task.

    The pipeline is constructed and executed synchronously; results are
    collected via a written parts prefix and read back, because a Beam
    pipeline cannot return a collection to its driver.
    """

    def __init__(
        self,
        pipeline_options: Mapping[str, Any],
        parts_dir: str,
        storage_options: dict | None = None,
    ) -> None:
        """Configure the Beam pipeline.

        ``pipeline_options`` is handed to Beam's ``PipelineOptions``
        verbatim (runner name, project, region, temp location, container
        image, ...), so supporting a further Beam runner is a configuration
        file and nothing else.

        Args:
            pipeline_options: Beam ``PipelineOptions`` keyword arguments.
            parts_dir: scratch directory for intermediate Beam output; must
                be reachable by both the driver and the Beam workers, which
                for any non-direct runner means a shared remote URI.
            storage_options: extra kwargs forwarded to fsspec.

        Raises:
            RuntimeError: naming the extra to install when Beam is unavailable.
            ValueError: when a non-direct runner is paired with a local
                ``parts_dir``.
        """
        if not optional._BEAM_AVAILABLE:
            raise RuntimeError(optional._BEAM_MISSING_MSG)

        options = dict(_plain(pipeline_options) or {})
        runner = options.get("runner")
        if not _is_direct_runner(runner) and _is_local_uri(parts_dir):
            raise ValueError(
                f"parts_dir {parts_dir!r} is a local location, but runner {runner!r} "
                "is not a direct runner — parts_dir must be reachable by the Beam "
                "workers as well as the driver, so it has to be a shared remote URI "
                "(for example gs://bucket/beam-parts)."
            )

        self.pipeline_options = options
        self.parts_dir = parts_dir
        self.storage_options = _plain(storage_options)

    def run_batches(
        self,
        batches: Sequence[Sequence[str]],
        images_root: str | None,
        masks_root: str | None,
        manifest_id: str,
        extractors: list[StatExtractor],
    ) -> list[BatchOutcome]:
        """Run every batch through one Beam pipeline and collect the outcomes.

        Args:
            batches: batches of image path sequences to process.
            images_root: root directory used to resolve mask mirror paths.
            masks_root: root directory of masks; None when unavailable.
            manifest_id: run identifier stamped on every produced record.
            extractors: list of StatExtractor callables.

        Returns:
            One :class:`~radiologist.etl.models.BatchOutcome` per input
            batch, in input order.

        Raises:
            RuntimeError: when the pipeline finishes in a failed state.
        """
        units = [[str(path) for path in batch] for batch in batches]
        if not units:
            return []

        parts_prefix = self._run_prefix("batches")
        self._run_pipeline(
            label="Batches",
            elements=list(enumerate(units)),
            fn=_beam_process_batch,
            images_root=images_root,
            masks_root=masks_root,
            manifest_id=manifest_id,
            extractors=extractors,
            parts_prefix=parts_prefix,
            storage_options=self.storage_options,
        )
        return [
            _batch_outcome_from_json(payload)
            for payload in _read_parts(parts_prefix, len(units), self.storage_options)
        ]

    def run_shards(self, jobs: Sequence[ShardJob]) -> list[ShardOutcome]:
        """Run every shard-writing job through one Beam pipeline.

        The build stage's counterpart to :meth:`run_batches` — each element
        writes its tar directly to its destination and emits its outcome
        under the run's parts prefix.

        Args:
            jobs: shard work units to run.

        Returns:
            One :class:`~radiologist.etl.models.ShardOutcome` per input job,
            in input order.

        Raises:
            RuntimeError: when the pipeline finishes in a failed state.
        """
        units = list(jobs)
        if not units:
            return []

        parts_prefix = self._run_prefix("shards")
        self._run_pipeline(
            label="Shards",
            elements=list(enumerate(units)),
            fn=_beam_write_shard,
            parts_prefix=parts_prefix,
            storage_options=self.storage_options,
        )
        return [
            _shard_outcome_from_json(payload)
            for payload in _read_parts(parts_prefix, len(units), self.storage_options)
        ]

    def _run_prefix(self, kind: str) -> str:
        """A parts prefix unique to this dispatch, so runs never read each other's."""
        return f"{self.parts_dir.rstrip('/')}/{kind}-{uuid.uuid4().hex}"

    def _run_pipeline(
        self,
        label: str,
        elements: list,
        fn: Any,
        **fn_kwargs: Any,
    ) -> None:
        """Build, run, and block on one pipeline applying ``fn`` to ``elements``.

        Raises:
            RuntimeError: when the pipeline does not finish in ``DONE``.
        """
        beam = optional.apache_beam
        options = optional.PipelineOptions(flags=[], **self.pipeline_options)

        pipeline = beam.Pipeline(options=options)
        _ = (
            pipeline
            | f"Create{label}" >> beam.Create(elements)
            | f"Process{label}" >> beam.Map(fn, **fn_kwargs)
        )

        result = pipeline.run()
        try:
            state = result.wait_until_finish()
        except Exception as exc:  # noqa: BLE001 - re-raised, named by state
            raise RuntimeError(
                f"Beam pipeline finished in state {_final_state(result)}: {exc}"
            ) from exc
        if state != optional.PipelineState.DONE:
            raise RuntimeError(f"Beam pipeline finished in state {state}, not DONE")
