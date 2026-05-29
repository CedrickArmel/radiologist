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

import hashlib
import json
import os
from pathlib import Path

import fsspec  # type: ignore[import-untyped]
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from prefect import flow, task
from prefect.artifacts import create_link_artifact, create_table_artifact
from prefect.cache_policies import INPUTS

from radiologist.etl.filters import filter_iqr, filter_lung_out_of_frame
from radiologist.etl.manifest import (
    JsonlWriter,
    ManifestRecord,
    ParquetWriter,
)
from radiologist.etl.processors import StatsProcessor
from radiologist.etl.shards import build_shards
from radiologist.etl.split import assign_split
from radiologist.etl.stats import StatExtractor, lung_asymmetry, make_haralick
from radiologist.utils.readers import ImageReader


def compute_run_id(
    cfg: DictConfig,
    source: str,
    storage_options: dict | None = None,
) -> str:
    """Compute a 16-char SHA-256 prefix run ID from config + data fingerprint.

    If cfg.run_label is set (non-None, non-empty), return it directly as the run ID.
    Otherwise hash: sorted(OmegaConf.to_container(cfg)) + file count + total bytes.

    Args:
        cfg: Hydra DictConfig with pipeline configuration.
        source: fsspec-compatible URI to the image root directory.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        A 16-character run ID string.
    """
    run_label = getattr(cfg, "run_label", None)
    if run_label:
        return str(run_label)

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    fs, root = fsspec.url_to_fs(source, **(storage_options or {}))
    all_files = fs.find(root)
    file_count = len(all_files)
    total_bytes = sum(fs.info(p)["size"] for p in all_files)
    fingerprint = json.dumps(
        {"cfg": cfg_dict, "file_count": file_count, "total_bytes": total_bytes},
        sort_keys=True,
    )
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


# ── Portable cores (NO Prefect imports needed here) ──────────────────────────


def _compute_stats(
    source: str,
    run_id: str,
    masks_root: str | None,
    extractors: list[StatExtractor],
    workers: int,
    artifact_dir: str,
    storage_options: dict | None = None,
) -> str:
    """Run stat extraction over the image collection and write a Parquet file.

    Args:
        source: fsspec-compatible URI to the image root directory.
        run_id: 16-char run identifier stamped on every ManifestRecord.
        masks_root: optional root directory for segmentation masks.
        extractors: list of StatExtractor callables.
        workers: number of worker processes.
        artifact_dir: local directory where the Parquet file is written.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        Path to the written Parquet file: ``{artifact_dir}/stats-{run_id}.parquet``.
    """
    reader = ImageReader(source, storage_options=storage_options)
    processor = StatsProcessor(extractors=extractors, workers=workers)
    records = processor.run(reader, manifest_id=run_id, masks_root=masks_root)
    dest = f"{artifact_dir}/stats-{run_id}.parquet"
    ParquetWriter().write(records, dest, storage_options=storage_options)
    return dest


def _apply_filters(
    parquet_path: str,
    iqr_columns: list[str],
    factor: float = 1.5,
    storage_options: dict | None = None,
) -> str:
    """Apply IQR and lung-out-of-frame filters, writing a new Parquet file.

    Args:
        parquet_path: path to the stats Parquet file.
        iqr_columns: column names to test for IQR outliers.
        factor: IQR multiplier for the fence. Default 1.5.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        Path to the filtered Parquet: same dir, with ``-filtered`` suffix.
    """
    df = pd.read_parquet(parquet_path)
    if iqr_columns:
        df = filter_iqr(df, iqr_columns, factor=factor)
    df = filter_lung_out_of_frame(df)
    stem = Path(parquet_path).stem
    out = str(Path(parquet_path).parent / f"{stem}-filtered.parquet")
    records = _df_to_records(df)
    ParquetWriter().write(records, out, storage_options=storage_options)
    return out


def _assign_splits(
    parquet_path: str,
    ratios: dict[str, float],
    storage_options: dict | None = None,
) -> str:
    """Assign train/val/test splits deterministically by filename hash.

    Args:
        parquet_path: path to the filtered Parquet file.
        ratios: mapping from split name to fraction.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        Path to the split Parquet: same dir, with ``-split`` replacing ``-filtered``.
    """
    df = pd.read_parquet(parquet_path)
    df["split"] = df["filename"].apply(lambda f: assign_split(f, ratios))
    p = Path(parquet_path)
    run_id_part = p.stem.replace("stats-", "").replace("-filtered", "")
    out = str(p.parent / f"stats-{run_id_part}-split.parquet")
    records = _df_to_records(df)
    ParquetWriter().write(records, out, storage_options=storage_options)
    return out


def _df_to_records(df: pd.DataFrame) -> list[ManifestRecord]:
    """Reconstruct ManifestRecord list from a flat DataFrame.

    Args:
        df: DataFrame with columns matching the ManifestRecord flat layout.

    Returns:
        List of ManifestRecord instances.
    """
    return [
        ManifestRecord.from_flat_dict(row._asdict())
        for row in df.itertuples(index=False)
    ]


def _write_jsonl(
    parquet_path: str,
    destination: str,
    storage_options: dict | None = None,
) -> str:
    """Write a JSONL manifest from a split Parquet file.

    Args:
        parquet_path: path to the split Parquet file.
        destination: output path for the JSONL manifest.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        The destination path (unchanged).
    """
    df = pd.read_parquet(parquet_path)
    records = _df_to_records(df)
    JsonlWriter().write(records, destination, storage_options=storage_options)
    return destination


# ── Prefect task shells ───────────────────────────────────────────────────────


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
    run_id = Path(parquet_path).stem.replace("stats-", "")
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
    run_id = Path(parquet_path).stem.replace("stats-", "").replace("-filtered", "")
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
    run_id = Path(parquet_path).stem.replace("stats-", "").replace("-split", "")
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
) -> str:
    """Prefect task: build WebDataset tar shards.

    Args:
        manifest_path: path to the JSONL manifest.
        shard_root: directory where shards are written.
        ratios: configured split ratios.
        shard_size: max samples per shard.
        start_shard_index: per-(split, label) shard index offset.

    Returns:
        Updated manifest_path.
    """
    out = build_shards(
        manifest_path=manifest_path,
        shard_root=shard_root,
        ratios=ratios,
        shard_size=shard_size,
        start_shard_index=start_shard_index,
    )
    run_id = Path(manifest_path).stem.split("-", 1)[1]
    report_path = str(Path(manifest_path).parent / f"split-report-{run_id}.json")
    with open(report_path) as f:
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


# ── Flow ──────────────────────────────────────────────────────────────────────


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
    return list(val) or None if val else None


@flow
def etl_flow(cfg: DictConfig) -> str:
    """Run the full ETL pipeline: stats → filter → split → manifest → (shards).

    Args:
        cfg: Hydra DictConfig with all pipeline parameters.

    Returns:
        Path to the final JSONL manifest file.
    """
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
        )

    return manifest_path
