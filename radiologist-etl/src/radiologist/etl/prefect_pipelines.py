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

"""Prefect flow and tasks orchestrating the end-to-end ETL pipeline."""

from __future__ import annotations

import functools
import json
from collections.abc import Sequence
from typing import Any

import fsspec  # type: ignore[import-untyped]
from omegaconf import DictConfig, OmegaConf

from radiologist.etl.optional import (
    _PREFECT_AVAILABLE,
    _PREFECT_IMPORT_ERROR,
    INPUTS,
    create_link_artifact,
    create_markdown_artifact,
    create_table_artifact,
    flow,
    task,
    unmapped,
)
from radiologist.utils import Logger

logger = Logger(name=__name__)

from radiologist.etl.assign import assign_splits as assign_splits_stage  # noqa: E402
from radiologist.etl.build import build_shards as build_shards_stage  # noqa: E402
from radiologist.etl.execution import (  # noqa: E402
    BatchMapper,
    ExecutionPlan,
    ShardMapper,
    resolve_execution,
)
from radiologist.etl.extract import extract as extract_stage  # noqa: E402
from radiologist.etl.models import (  # noqa: E402
    AssignSplitResult,
    BatchOutcome,
    BuildResult,
    ExtractResult,
    ShardJob,
    ShardOutcome,
)
from radiologist.etl.processors import process_batch  # noqa: E402
from radiologist.etl.shards import write_shard  # noqa: E402
from radiologist.etl.split import SplitRatios  # noqa: E402
from radiologist.etl.stats import (  # noqa: E402
    StatExtractor,
    lung_asymmetry,
    make_haralick,
)


def _haralick_list(cfg_node: object, key: str) -> list | None:
    """Extract a list value from a haralick config node.

    Args:
        cfg_node: dict or OmegaConf DictConfig holding haralick settings.
        key: key to retrieve (e.g. "features", "distances", "angles").

    Returns:
        A non-empty list or None.
    """
    if isinstance(cfg_node, dict):
        val = cfg_node.get(key)
    else:
        val = OmegaConf.select(cfg_node, key)  # type: ignore[arg-type]
    return (list(val) or None) if val else None


@task(cache_policy=INPUTS)
def extract_batch_task(
    paths: list[str],
    images_root: str | None,
    masks_root: str | None,
    manifest_id: str,
    extractors: list[StatExtractor],
    storage_options: dict | None = None,
) -> BatchOutcome:
    """Prefect task: process one batch of image paths for the extract stage.

    Args:
        paths: image paths to process in this batch.
        images_root: root directory used to resolve mask mirror paths.
        masks_root: root directory of masks; None when masks are unavailable.
        manifest_id: run identifier stamped on every produced record.
        extractors: list of StatExtractor callables.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        A :class:`~radiologist.etl.models.BatchOutcome` for this batch.
    """
    return process_batch(
        paths,
        images_root=images_root,
        masks_root=masks_root,
        manifest_id=manifest_id,
        extractors=extractors,
        storage_options=storage_options,
    )


@task(cache_policy=INPUTS)
def write_shard_task(
    job: ShardJob,
    storage_options: dict | None = None,
) -> ShardOutcome:
    """Prefect task: write one WebDataset tar shard for the build stage.

    Args:
        job: the shard's work unit.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        A :class:`~radiologist.etl.models.ShardOutcome` describing the write.
    """
    return write_shard(job, storage_options=storage_options)


def with_task_runner(flow_obj: Any, plan: ExecutionPlan) -> Any:
    """Attach an :class:`~radiologist.etl.execution.ExecutionPlan`'s task runner to a flow.

    Args:
        flow_obj: a Prefect flow object.
        plan: the resolved execution plan.

    Returns:
        ``flow_obj.with_options(task_runner=plan.task_runner)`` when a task
        runner is present and prefect is installed; ``flow_obj`` unchanged
        otherwise.
    """
    if _PREFECT_AVAILABLE and plan.task_runner is not None:
        return flow_obj.with_options(task_runner=plan.task_runner)
    return flow_obj


def _storage_options_from_cfg(cfg: DictConfig) -> dict | None:
    """Pull a plain ``dict`` out of ``cfg.storage_options`` (or ``None``)."""
    raw = (
        OmegaConf.to_container(cfg.storage_options)
        if OmegaConf.select(cfg, "storage_options") is not None
        else None
    )
    return dict(raw) if isinstance(raw, dict) else None


def _ordered_ratios(cfg: DictConfig) -> SplitRatios | None:
    """Pull an ordered ``[(name, fraction), ...]`` sequence out of ``cfg.split_ratios``."""
    raw = OmegaConf.select(cfg, "split_ratios")
    if raw is None:
        return None
    pairs = raw if isinstance(raw, list) else OmegaConf.to_container(raw)
    if not isinstance(pairs, list):
        raise ValueError(f"split_ratios must resolve to a list of pairs, got {raw!r}")
    return tuple((str(name), float(fraction)) for name, fraction in pairs)


def _extract_batch_mapper(
    plan: ExecutionPlan,
    images_root: str | None,
    masks_root: str | None,
    extractors: list[StatExtractor],
    storage_options: dict | None,
) -> BatchMapper | None:
    """Build the extract stage's mapper from the resolved plan.

    Returns ``None`` to defer to :func:`~radiologist.etl.extract.extract`'s
    own local default mapper.

    ``manifest_id`` cannot be known here — ``extract()`` only computes the
    run id after this mapper is built — so every branch below binds a
    placeholder; ``extract()`` unconditionally restamps every record's
    ``manifest_id`` to the real run id once collected, regardless of which
    mapper produced them.
    """
    if plan.beam is not None:
        return functools.partial(
            plan.beam.run_batches,
            images_root=images_root,
            masks_root=masks_root,
            manifest_id="",
            extractors=extractors,
        )
    if plan.task_runner is not None:

        def _mapped(batches: Sequence[Sequence[str]]) -> list[BatchOutcome]:
            futures = extract_batch_task.map(
                batches,
                images_root=unmapped(images_root),
                masks_root=unmapped(masks_root),
                manifest_id=unmapped(""),
                extractors=unmapped(extractors),
                storage_options=unmapped(storage_options),
            )
            return list(futures.result())

        return _mapped
    return None


def _shard_mapper(
    plan: ExecutionPlan,
    storage_options: dict | None,
) -> ShardMapper | None:
    """Build the build stage's mapper from the resolved plan.

    Returns ``None`` to defer to
    :func:`~radiologist.etl.build.build_shards`'s own local default mapper.
    """
    if plan.beam is not None:
        return plan.beam.run_shards
    if plan.task_runner is not None:

        def _mapped(jobs: Sequence[ShardJob]) -> list[ShardOutcome]:
            futures = write_shard_task.map(
                jobs, storage_options=unmapped(storage_options)
            )
            return list(futures.result())

        return _mapped
    return None


@flow(name="etl-extract")
def extract_flow(
    cfg: DictConfig, execution: ExecutionPlan | None = None
) -> ExtractResult:
    """Prefect flow wrapping the extract stage.

    Args:
        cfg: Hydra DictConfig with the extract stage's parameters.
        execution: resolved execution plan; defaults to a local plan.

    Returns:
        An :class:`~radiologist.etl.models.ExtractResult` describing the run.
    """
    if not _PREFECT_AVAILABLE:
        logger.warning(
            f"{_PREFECT_IMPORT_ERROR}: prefect is missing. This run will not be recorded!"
        )

    batch_size = int(cfg.batch_size) if OmegaConf.select(cfg, "batch_size") else None
    plan = (
        execution
        if execution is not None
        else resolve_execution(OmegaConf.select(cfg, "runner"), batch_size=batch_size)
    )

    create_markdown_artifact(
        key="extract-config",
        markdown=(
            f"```yaml\nrunner_family: {plan.family}\n"
            f"{OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)}\n```"
        ),
    )

    storage_options = _storage_options_from_cfg(cfg)
    images_root = OmegaConf.select(cfg, "images_root")
    masks_root = OmegaConf.select(cfg, "masks_root")

    haralick_cfg = OmegaConf.select(cfg, "haralick") or {}
    extractors: list[StatExtractor] = [
        make_haralick(
            features=_haralick_list(haralick_cfg, "features"),
            distances=_haralick_list(haralick_cfg, "distances"),
            angles=_haralick_list(haralick_cfg, "angles"),
        ),
        lung_asymmetry,
    ]

    mapper = _extract_batch_mapper(
        plan, images_root, masks_root, extractors, storage_options
    )

    result = extract_stage(
        file_list=cfg.file_list,
        destination=cfg.destination,
        images_root=images_root,
        masks_root=masks_root,
        extractors=extractors,
        iqr_columns=(
            list(cfg.iqr_columns)
            if OmegaConf.select(cfg, "iqr_columns") is not None
            else None
        ),
        iqr_factor=(
            float(cfg.iqr_factor)
            if OmegaConf.select(cfg, "iqr_factor") is not None
            else 1.5
        ),
        workers=int(cfg.workers) if OmegaConf.select(cfg, "workers") else None,
        batch_size=plan.batch_size,
        max_failure_rate=(
            float(cfg.max_failure_rate)
            if OmegaConf.select(cfg, "max_failure_rate") is not None
            else 0.0
        ),
        run_label=OmegaConf.select(cfg, "run_label"),
        mapper=mapper,
        storage_options=storage_options,
    )

    create_link_artifact(
        link=result.manifest_path,
        key=f"extract-{result.run_id}",
        description=(
            f"Extract manifest for run {result.run_id}: "
            f"{result.total} total, {result.succeeded} succeeded, "
            f"{result.failed} failed, {result.excluded} excluded."
        ),
    )
    return result


@flow(name="etl-assign-split")
def assign_split_flow(cfg: DictConfig) -> AssignSplitResult:
    """Prefect flow wrapping the assign-split stage.

    Args:
        cfg: Hydra DictConfig with the assign-split stage's parameters.

    Returns:
        An :class:`~radiologist.etl.models.AssignSplitResult` describing the run.
    """
    if not _PREFECT_AVAILABLE:
        logger.warning(
            f"{_PREFECT_IMPORT_ERROR}: prefect is missing. This run will not be recorded!"
        )

    create_markdown_artifact(
        key="assign-split-config",
        markdown=f"```yaml\n{OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)}\n```",
    )

    result = assign_splits_stage(
        manifests_dir=cfg.manifests_dir,
        destination=cfg.destination,
        ratios=_ordered_ratios(cfg),
        run_label=OmegaConf.select(cfg, "run_label"),
        storage_options=_storage_options_from_cfg(cfg),
    )

    create_link_artifact(
        link=result.split_manifest_path,
        key=f"assign-split-{result.run_id}",
        description=(
            f"Split manifest for run {result.run_id}: "
            f"{result.source_manifest_count} source manifest(s), "
            f"{result.duplicate_count} duplicate(s) dropped, "
            f"counts_by_split={result.counts_by_split}."
        ),
    )
    return result


@flow(name="etl-build")
def build_flow(cfg: DictConfig, execution: ExecutionPlan | None = None) -> BuildResult:
    """Prefect flow wrapping the build stage.

    Args:
        cfg: Hydra DictConfig with the build stage's parameters.
        execution: resolved execution plan; defaults to a local plan.

    Returns:
        A :class:`~radiologist.etl.models.BuildResult` describing the run.
    """
    if not _PREFECT_AVAILABLE:
        logger.warning(
            f"{_PREFECT_IMPORT_ERROR}: prefect is missing. This run will not be recorded!"
        )

    plan = (
        execution
        if execution is not None
        else resolve_execution(OmegaConf.select(cfg, "runner"))
    )

    create_markdown_artifact(
        key="build-config",
        markdown=(
            f"```yaml\nrunner_family: {plan.family}\n"
            f"{OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)}\n```"
        ),
    )

    storage_options = _storage_options_from_cfg(cfg)
    mapper = _shard_mapper(plan, storage_options)

    result = build_shards_stage(
        split_manifest_path=cfg.split_manifest,
        shard_root=cfg.shard_root,
        shard_size=int(cfg.shard_size),
        ratios=_ordered_ratios(cfg),
        workers=int(cfg.workers) if OmegaConf.select(cfg, "workers") else None,
        run_label=OmegaConf.select(cfg, "run_label"),
        mapper=mapper,
        storage_options=storage_options,
    )

    opts = storage_options or {}
    fs_r, rpath = fsspec.url_to_fs(result.report_path, **opts)
    with fs_r.open(rpath, "rt", encoding="utf-8") as f:
        report = json.load(f)
    observed = report.get("observed", {})
    rows = [
        {"label": label, **split_counts} for label, split_counts in observed.items()
    ]
    create_table_artifact(
        table=rows,
        key=f"build-{result.run_id}",
        description=f"Shard split report for run {result.run_id}",
    )

    create_link_artifact(
        link=result.output_dir,
        key=f"build-{result.run_id}",
        description=(
            f"Build output for run {result.run_id}: "
            f"{result.shard_count} shard(s), {result.record_count} record(s)."
        ),
    )
    return result


def run_extract(cfg: DictConfig) -> ExtractResult:
    """Resolve the runner config, attach the task runner, and run :func:`extract_flow`.

    Args:
        cfg: Hydra DictConfig with the extract stage's parameters.

    Returns:
        An :class:`~radiologist.etl.models.ExtractResult` describing the run.
    """
    batch_size = int(cfg.batch_size) if OmegaConf.select(cfg, "batch_size") else None
    plan = resolve_execution(OmegaConf.select(cfg, "runner"), batch_size=batch_size)
    flow_to_run = with_task_runner(extract_flow, plan)
    return flow_to_run(cfg, execution=plan)


def run_assign_split(cfg: DictConfig) -> AssignSplitResult:
    """Run :func:`assign_split_flow` (this stage never uses a runner).

    Args:
        cfg: Hydra DictConfig with the assign-split stage's parameters.

    Returns:
        An :class:`~radiologist.etl.models.AssignSplitResult` describing the run.
    """
    return assign_split_flow(cfg)


def run_build(cfg: DictConfig) -> BuildResult:
    """Resolve the runner config, attach the task runner, and run :func:`build_flow`.

    Args:
        cfg: Hydra DictConfig with the build stage's parameters.

    Returns:
        A :class:`~radiologist.etl.models.BuildResult` describing the run.
    """
    plan = resolve_execution(OmegaConf.select(cfg, "runner"))
    flow_to_run = with_task_runner(build_flow, plan)
    return flow_to_run(cfg, execution=plan)
