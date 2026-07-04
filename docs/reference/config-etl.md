# ETL (Hydra) Configuration Reference

The ETL pipeline is a [Prefect](https://www.prefect.io/) flow (`radiologist.etl.prefect_pipelines:etl_flow`)
composed via Hydra:

```python
@hydra.main(config_path="conf", config_name="etl", version_base=None)
def main(cfg: DictConfig) -> None:
    ...
```

Unlike the training and inference entry points, the ETL flow has no interactive
`--help` surface — it is invoked as a script and every parameter is overridden
via standard Hydra CLI syntax (`key=value`). The full default config lives at
`radiologist-etl/src/radiologist/etl/conf/etl.yaml`.

```bash
cd radiologist-etl
uv run --active python -m radiologist.etl.prefect_pipelines \
    source=gs://bucket/raw_chest_x_ray_data/images \
    destination=gs://bucket/manifests/ \
    shard_root=gs://bucket/shards/
```

## Pipeline parameters

| Key | Default | Description |
|---|---|---|
| `source` | GCS URI | Root directory of raw labelled images (local path or fsspec URI). |
| `masks_root` | GCS URI | Root directory of segmentation masks, mirrored path-for-path under `source`. Used by `lung_out_of_frame` / `lung_asymmetry`; set to `null` to skip mask-dependent features. |
| `destination` | GCS URI | Output directory for the JSONL manifest (`manifest-{run_id}.jsonl`). |
| `artifact_dir` | GCS URI | Directory where intermediate Parquet artifacts (`stats-*.parquet`) are written. |
| `storage_options` | `null` | Extra kwargs forwarded to `fsspec.url_to_fs` for every remote read/write (e.g. GCS credentials). |
| `workers` | `null` (all CPUs) | Number of worker processes for stat extraction; `null` resolves to `os.cpu_count()`. |
| `run_label` | `null` | Optional label folded into the run-ID hash so intentional re-runs on unchanged data get a new `run_id` instead of colliding. |

## Quality filtering

| Key | Default | Description |
|---|---|---|
| `iqr_columns` | `["haralick_mean"]` | Stat columns tested for IQR outliers via `filter_iqr`. Set to `[]` to disable IQR filtering. |
| `iqr_factor` | `1.5` | IQR multiplier defining the outlier fence (`Q1 - factor*IQR`, `Q3 + factor*IQR`). |

## Haralick GLCM features

`cfg.haralick` is passed to `make_haralick(...)` to build the texture-feature extractor:

| Key | Default | Description |
|---|---|---|
| `haralick.features` | `["mean"]` | Subset of `HARALICK_PROPERTIES` to compute (`mean`, `std`, `entropy`, `contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`, `ASM`); `null` computes all nine. |
| `haralick.distances` | `[1]` | Pixel-pair distances for the GLCM; `null` defaults to `[1]`. |
| `haralick.angles` | `null` | Angles in radians for the GLCM; `null` defaults to `[0, π/4, π/2, 3π/4]`. |

## Splitting and sharding

| Key | Default | Description |
|---|---|---|
| `split_ratios.train` / `.val` / `.test` | `0.70` / `0.15` / `0.15` | Deterministic split fractions consumed by `assign_split`; must sum to `1.0`. |
| `build_shards` | `true` | Whether to produce WebDataset tar shards after the manifest is written. |
| `shard_root` | GCS URI | Directory where `{split}/{label}/{split}-{label}-{idx:06d}.tar` shards are written. |
| `shard_size` | `1000` | Max samples per tar shard. |

## Resuming a prior run

Each stage is independently resumable via Prefect task caching and the
`resume_from_*` flags, which let a run start from a previously-written
intermediate artifact instead of recomputing it:

| Key | Default | Description |
|---|---|---|
| `resume_from_parquet` | `null` | Path to an existing `stats-*.parquet`; skips stat extraction. |
| `resume_from_filtered` | `null` | Path to an existing `stats-*-filtered.parquet`; skips stat extraction and filtering. |
| `resume_from_split` | `null` | Path to an existing `stats-*-split.parquet`; skips through split assignment. |
| `resume_from_manifest` | `null` | Path to an existing `manifest-*.jsonl`; skips straight to sharding. |

```bash
uv run --active python -m radiologist.etl.prefect_pipelines \
    resume_from_manifest=true \
    destination=gs://bucket/manifests/ \
    build_shards=true
```

See `radiologist-etl/README.md` for the full pipeline-stage diagram and
programmatic usage examples.
