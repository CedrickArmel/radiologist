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
from pathlib import Path

import fsspec  # type: ignore[import-untyped]
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from radiologist.etl.filters import filter_iqr, filter_lung_out_of_frame
from radiologist.etl.manifest import (
    JsonlWriter,
    ManifestRecord,
    ParquetWriter,
)
from radiologist.etl.processors import StatsProcessor
from radiologist.etl.shards import build_shards
from radiologist.etl.split import assign_split
from radiologist.etl.stats import StatExtractor
from radiologist.utils.readers import ImageReader


def compute_run_id(
    cfg: DictConfig,
    source: str,
    storage_options: dict | None = None,
) -> str:
    """Compute a 16-char SHA-256 prefix run ID from config + data fingerprint.

    Hashes: sorted(OmegaConf.to_container(cfg)) + file count + total bytes +
    run_label (None when not set).  Including run_label in the hash means two
    calls with the same label on the same data return the same ID (idempotent),
    while changing the label forces a new ID without silently overwriting prior
    artifacts.

    Args:
        cfg: Hydra DictConfig with pipeline configuration.
        source: fsspec-compatible URI to the image root directory.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        A 16-character run ID string.
    """
    run_label = getattr(cfg, "run_label", None)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    fs, root = fsspec.url_to_fs(source, **(storage_options or {}))
    all_files = fs.find(root)
    file_count = len(all_files)
    total_bytes = sum(fs.info(p)["size"] for p in all_files)
    fingerprint = json.dumps(
        {
            "cfg": cfg_dict,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "run_label": run_label,
        },
        sort_keys=True,
    )
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:16]


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
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
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


def _build_shards(
    manifest_path: str,
    shard_root: str,
    ratios: dict[str, float],
    shard_size: int = 1000,
    start_shard_index: dict[tuple[str, str], int] | None = None,
    storage_options: dict | None = None,
) -> str:
    """Portable core: build WebDataset tar shards.

    Args:
        manifest_path: path to the JSONL manifest.
        shard_root: directory where shards are written.
        ratios: configured split ratios.
        shard_size: max samples per shard.
        start_shard_index: per-(split, label) shard index offset.
        storage_options: extra kwargs forwarded to fsspec.

    Returns:
        Updated manifest path.
    """
    return build_shards(
        manifest_path=manifest_path,
        shard_root=shard_root,
        ratios=ratios,
        shard_size=shard_size,
        start_shard_index=start_shard_index,
        storage_options=storage_options,
    )


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
