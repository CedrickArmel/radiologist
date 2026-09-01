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

import json
import os
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
)
from radiologist.utils import Logger

logger = Logger(name=__name__)

from radiologist.etl.execution import ExecutionPlan  # noqa: E402
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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


@flow(name="etl-assign-split")
def assign_split_flow(cfg: DictConfig) -> AssignSplitResult:
    """Prefect flow wrapping the assign-split stage.

    Args:
        cfg: Hydra DictConfig with the assign-split stage's parameters.

    Returns:
        An :class:`~radiologist.etl.models.AssignSplitResult` describing the run.
    """
    raise NotImplementedError


@flow(name="etl-build")
def build_flow(cfg: DictConfig, execution: ExecutionPlan | None = None) -> BuildResult:
    """Prefect flow wrapping the build stage.

    Args:
        cfg: Hydra DictConfig with the build stage's parameters.
        execution: resolved execution plan; defaults to a local plan.

    Returns:
        A :class:`~radiologist.etl.models.BuildResult` describing the run.
    """
    raise NotImplementedError


def run_extract(cfg: DictConfig) -> ExtractResult:
    """Resolve the runner config, attach the task runner, and run :func:`extract_flow`.

    Args:
        cfg: Hydra DictConfig with the extract stage's parameters.

    Returns:
        An :class:`~radiologist.etl.models.ExtractResult` describing the run.
    """
    raise NotImplementedError


def run_assign_split(cfg: DictConfig) -> AssignSplitResult:
    """Run :func:`assign_split_flow` (this stage never uses a runner).

    Args:
        cfg: Hydra DictConfig with the assign-split stage's parameters.

    Returns:
        An :class:`~radiologist.etl.models.AssignSplitResult` describing the run.
    """
    raise NotImplementedError


def run_build(cfg: DictConfig) -> BuildResult:
    """Resolve the runner config, attach the task runner, and run :func:`build_flow`.

    Args:
        cfg: Hydra DictConfig with the build stage's parameters.

    Returns:
        A :class:`~radiologist.etl.models.BuildResult` describing the run.
    """
    raise NotImplementedError
