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
import os
from collections.abc import Sequence
from typing import Any

import fsspec  # type: ignore[import-untyped]
from omegaconf import DictConfig, OmegaConf

import radiologist.utils.filesystem as fst
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
    EtlResult,
    ExtractResult,
    ShardJob,
    ShardOutcome,
)
from radiologist.etl.ops import (  # noqa: E402
    _apply_filters,
    _assign_splits,
    _build_shards,
    _compute_stats,
    _write_jsonl,
    compute_run_id,
)
from radiologist.etl.processors import process_batch  # noqa: E402
from radiologist.etl.shards import write_shard  # noqa: E402
from radiologist.etl.split import SplitRatios  # noqa: E402
from radiologist.etl.stats import (  # noqa: E402
    StatExtractor,
    lung_asymmetry,
    make_haralick,
)


@task(cache_policy=INPUTS)
def compute_stats_task(
    source: str,
    run_id: str,
    masks_root: str | None,
    extractors: list[StatExtractor],
    workers: int,
    artifact_dir: str,
    storage_options: dict | None = None,
) -> str:
    """Prefect task: run stat extraction and link artifact.

    Args:
        source: fsspec-compatible URI to the image root directory.
        run_id: run identifier.
        masks_root: optional mask root directory.
        extractors: list of StatExtractor callables.
        workers: number of worker processes.
        artifact_dir: directory for intermediate Parquet files.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        Path to the written Parquet file.
    """
    dest = _compute_stats(
        source=source,
        run_id=run_id,
        masks_root=masks_root,
        extractors=extractors,
        workers=workers,
        artifact_dir=artifact_dir,
        storage_options=storage_options,
    )
    create_link_artifact(
        link=dest,
        key=f"stats-{run_id}",
        description=f"Raw stats Parquet for run {run_id}",
    )
    return dest


@task(cache_policy=INPUTS)
def apply_filters_task(
    parquet_path: str,
    iqr_columns: list[str],
    factor: float = 1.5,
    storage_options: dict | None = None,
) -> str:
    """Prefect task: apply IQR and lung-out-of-frame filters.

    Args:
        parquet_path: path to the stats Parquet file.
        iqr_columns: column names to test for IQR outliers.
        factor: IQR multiplier.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        Path to the filtered Parquet file.
    """
    out = _apply_filters(
        parquet_path=parquet_path,
        iqr_columns=iqr_columns,
        factor=factor,
        storage_options=storage_options,
    )
    run_id = fst.pathstem(parquet_path).replace("stats-", "")
    create_link_artifact(
        link=out,
        key=f"stats-{run_id}-filtered",
        description=f"Filtered stats Parquet for run {run_id}",
    )
    return out


@task(cache_policy=INPUTS)
def assign_splits_task(
    parquet_path: str,
    ratios: dict[str, float],
    storage_options: dict | None = None,
) -> str:
    """Prefect task: assign train/val/test splits.

    Args:
        parquet_path: path to the filtered Parquet file.
        ratios: mapping from split name to fraction.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        Path to the split Parquet file.
    """
    out = _assign_splits(
        parquet_path=parquet_path,
        ratios=ratios,
        storage_options=storage_options,
    )
    run_id = fst.pathstem(parquet_path).replace("stats-", "").replace("-filtered", "")
    create_link_artifact(
        link=out,
        key=f"stats-{run_id}-split",
        description=f"Split-assigned stats Parquet for run {run_id}",
    )
    return out


@task(cache_policy=INPUTS)
def write_jsonl_task(
    parquet_path: str,
    destination: str,
    storage_options: dict | None = None,
) -> str:
    """Prefect task: write the JSONL manifest.

    Args:
        parquet_path: path to the split Parquet file.
        destination: output path for the JSONL manifest.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        The destination path.
    """
    out = _write_jsonl(
        parquet_path=parquet_path,
        destination=destination,
        storage_options=storage_options,
    )
    run_id = fst.pathstem(parquet_path).replace("stats-", "").replace("-split", "")
    create_link_artifact(
        link=out,
        key=f"manifest-{run_id}",
        description=f"JSONL manifest for run {run_id}",
    )
    return out


@task(cache_policy=INPUTS)
def build_shards_task(
    manifest_path: str,
    shard_root: str,
    ratios: dict[str, float],
    shard_size: int = 1000,
    start_shard_index: dict[tuple[str, str], int] | None = None,
    storage_options: dict | None = None,
) -> str:
    """Prefect task: build WebDataset tar shards.

    Args:
        manifest_path: path to the JSONL manifest.
        shard_root: directory where shards are written.
        ratios: configured split ratios.
        shard_size: max samples per shard.
        start_shard_index: per-(split, label) shard index offset.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        Updated manifest_path.
    """
    out = _build_shards(
        manifest_path=manifest_path,
        shard_root=shard_root,
        ratios=ratios,
        shard_size=shard_size,
        start_shard_index=start_shard_index,
        storage_options=storage_options,
    )
    manifest_stem = fst.pathstem(manifest_path)
    run_id = manifest_stem.split("-", 1)[1] if "-" in manifest_stem else manifest_stem
    manifest_parent = manifest_path.rsplit("/", 1)[0]
    report_path = f"{manifest_parent}/split-report-{run_id}.json"
    opts = storage_options or {}
    fs_r, rpath = fsspec.url_to_fs(report_path, **opts)
    with fs_r.open(rpath, "rt", encoding="utf-8") as f:
        report = json.load(f)
    observed = report.get("observed", {})
    rows = [
        {"label": label, **split_counts} for label, split_counts in observed.items()
    ]
    create_table_artifact(
        table=rows,
        key=f"split-report-{run_id}",
        description=f"Shard split report for run {run_id}",
    )
    return out


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


@flow
def etl_flow(cfg: DictConfig) -> EtlResult:
    """Run the full ETL pipeline: stats → filter → split → manifest → (shards).

    Args:
        cfg: Hydra DictConfig with all pipeline parameters.

    Returns:
        An :class:`~radiologist.etl.models.EtlResult` carrying the run id
        used to name every artifact and the path to the final JSONL
        manifest file.
    """
    if not _PREFECT_AVAILABLE:
        logger.warning(
            f"{_PREFECT_IMPORT_ERROR}: prefect is missing. This flow will not be recorded!"
        )

    create_markdown_artifact(
        key="etlconfig",
        markdown=f"```yaml\n{OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)}\n```",
    )

    _so_raw = (
        OmegaConf.to_container(cfg.storage_options)
        if OmegaConf.select(cfg, "storage_options") is not None
        else None
    )
    storage_options: dict | None = dict(_so_raw) if isinstance(_so_raw, dict) else None
    source = cfg.source
    run_id = compute_run_id(cfg, source, storage_options=storage_options)

    haralick_cfg = OmegaConf.select(cfg, "haralick") or {}
    features = _haralick_list(haralick_cfg, "features")
    distances = _haralick_list(haralick_cfg, "distances")
    angles = _haralick_list(haralick_cfg, "angles")

    extractor = make_haralick(features=features, distances=distances, angles=angles)
    extractors: list[StatExtractor] = [extractor, lung_asymmetry]
    workers: int = int(cfg.workers) if cfg.workers else (os.cpu_count() or 1)

    manifest_dest = f"{cfg.destination}/manifest-{run_id}.jsonl"

    resume_parquet = OmegaConf.select(cfg, "resume_from_parquet")
    resume_filtered = OmegaConf.select(cfg, "resume_from_filtered")
    resume_split = OmegaConf.select(cfg, "resume_from_split")
    resume_manifest = OmegaConf.select(cfg, "resume_from_manifest")

    parquet_path: str = resume_parquet or compute_stats_task(
        source=source,
        run_id=run_id,
        masks_root=OmegaConf.select(cfg, "masks_root"),
        extractors=extractors,
        workers=workers,
        artifact_dir=cfg.artifact_dir,
        storage_options=storage_options,
    )
    filtered_path: str = resume_filtered or apply_filters_task(
        parquet_path=parquet_path,
        iqr_columns=list(cfg.iqr_columns) if cfg.iqr_columns else [],
        factor=float(cfg.iqr_factor),
    )
    split_path: str = resume_split or assign_splits_task(
        parquet_path=filtered_path,
        ratios=OmegaConf.to_container(cfg.split_ratios),  # type: ignore[arg-type]
    )
    manifest_path: str = resume_manifest or write_jsonl_task(
        parquet_path=split_path,
        destination=manifest_dest,
        storage_options=storage_options,
    )

    if cfg.build_shards:
        manifest_path = build_shards_task(
            manifest_path=manifest_path,
            shard_root=cfg.shard_root,
            ratios=OmegaConf.to_container(cfg.split_ratios),  # type: ignore[arg-type]
            shard_size=int(cfg.shard_size),
            storage_options=storage_options,
        )

    return EtlResult(run_id=run_id, manifest_path=manifest_path)


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
