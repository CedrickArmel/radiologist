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

import fsspec  # type: ignore[import-untyped]
import hydra  # type: ignore[import-untyped]
from omegaconf import DictConfig, OmegaConf

import radiologist.utils.filesystem as fst
from radiologist.utils import Logger

logger = Logger(name=__name__)

try:
    from prefect import flow, task
    from prefect.artifacts import (
        create_link_artifact,
        create_markdown_artifact,
        create_table_artifact,
    )
    from prefect.cache_policies import INPUTS

    _PREFECT_AVAILABLE = True
except ImportError as ex:  # pragma: no cover

    def flow(fn=None, **_):  # type: ignore[misc, no-redef]
        """No-op stand-in for ``prefect.flow`` when prefect is not installed."""
        return fn if fn is not None else (lambda f: f)

    def task(fn=None, **_):  # type: ignore[misc, no-redef]
        """No-op stand-in for ``prefect.task`` when prefect is not installed."""
        return fn if fn is not None else (lambda f: f)

    def create_link_artifact(**_):  # type: ignore[misc, no-redef]
        """No-op stand-in for ``prefect.artifacts.create_link_artifact``."""

    def create_markdown_artifact(**_):
        """No-op stand-in for ``prefect.artifacts.create_markdown_artifact``."""

    def create_table_artifact(**_):  # type: ignore[misc, no-redef]
        """No-op stand-in for ``prefect.artifacts.create_table_artifact``."""

    _PREFECT_AVAILABLE = False
    _PREFECT_IMPORT_ERROR: str = str(ex)
    INPUTS = None  # type: ignore[assignment]

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
def etl_flow(cfg: DictConfig) -> str:
    """Run the full ETL pipeline: stats → filter → split → manifest → (shards).

    Args:
        cfg: Hydra DictConfig with all pipeline parameters.

    Returns:
        Path to the final JSONL manifest file.
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

    return manifest_path


@hydra.main(config_path="conf", config_name="etl", version_base=None)
def main(cfg: DictConfig) -> None:
    """CLI entry point: run the full ETL pipeline from a Hydra config.

    Args:
        cfg: Hydra DictConfig populated from conf/etl.yaml and CLI overrides.
    """
    etl_flow(cfg)


if __name__ == "__main__":
    main()
