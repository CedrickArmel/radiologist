# radiologist-etl

[![ci](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml/badge.svg)](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CedrickArmel/radiologist/branch/main/graph/badge.svg)](https://codecov.io/gh/CedrickArmel/radiologist)
[![PyPI](https://img.shields.io/pypi/v/radiologist-etl)](https://pypi.org/project/radiologist-etl/)
![tested on](https://img.shields.io/badge/tested%20on-ubuntu--latest%20%7C%20python%203.10-blue)

Data preparation pipeline. Transforms a raw folder of labelled chest X-ray images (local or GCS) into streaming-ready WebDataset tar shards, producing a deterministic manifest and quality-filtered dataset.

## Business context

Raw clinical X-ray archives are large, heterogeneous, and noisy. Images may be cropped so severely that the lungs fall outside the frame, or they may contain GLCM texture patterns that are statistical outliers relative to the rest of the cohort. Training on such images degrades model accuracy and reproducibility.

`radiologist-etl` solves this before any model ever sees the data:

- **Quality gate** — Haralick GLCM features and lung-boundary checks flag images that should not enter training.
- **Deterministic splits** — every image is deterministically assigned to `train`, `val`, or `test` based on a hash of its filename, so reruns always produce the same split without storing split state.
- **Streaming format** — output shards (`WebDataset` tars) stream efficiently from GCS during training without downloading the full dataset.
- **Reproducibility** — a content-addressed run ID (SHA-256 over config + dataset fingerprint) makes each ETL run traceable.

## Pipeline stages

```
Raw images (GCS or local)
  └─ 1. StatsProcessor      → stats-{run_id}.parquet        (Haralick GLCM + lung asymmetry)
  └─ 2. filter_iqr           → stats-{run_id}-filtered.parquet  (IQR outlier removal)
       filter_lung_out_of_frame
  └─ 3. assign_split         → stats-{run_id}-split.parquet  (MD5-deterministic split)
  └─ 4. write_jsonl          → manifest-{run_id}.jsonl        (canonical manifest)
  └─ 5. build_shards         → {shard_root}/{split}/{label}/{split}-{label}-{idx:06d}.tar
```

Each stage is independently resumable via Prefect task caching and the `resume_from_*` config flags.

## Key classes and functions

### Manifest (`radiologist.etl.manifest`)

`ManifestRecord` is the canonical data row throughout the pipeline. Each image becomes one record.

```python
@dataclass
class ManifestRecord:
    manifest_id: str         # unique per run
    path: str                # original fsspec URI
    filename: str            # basename used for split assignment
    label: str               # raw class label from folder name
    split: str               # "train" | "val" | "test"
    stats: dict              # flat float dict of extracted features
    lung_out_of_frame: bool
    excluded: bool
    exclusion_reason: str    # pipe-joined list of reasons
    shard: str               # path to the tar shard (set by build_shards)
```

`ParquetWriter` and `JsonlWriter` serialise records to Parquet and JSONL respectively. `records_reader` deserialises a JSONL manifest back to a list of `ManifestRecord`.

### Feature extraction (`radiologist.etl.stats`)

```python
from radiologist.etl import make_haralick, lung_asymmetry

extractor = make_haralick(
    features=["mean", "std", "entropy", "contrast", "homogeneity"],
    distances=[1],
    angles=[0, 0.785, 1.571, 2.356],
)
# extractor(image, metadata, mask) -> dict[str, float]
```

`make_haralick` returns a `StatExtractor` (a `Protocol`-typed callable). Any function with signature `(image, metadata, mask?) -> dict[str, float]` can be plugged into `StatsProcessor` in its place.

`lung_asymmetry(image, metadata, mask)` — returns `asymmetry_ratio` and `asymmetry_diff` from a lung segmentation mask.

### Quality filtering (`radiologist.etl.filters`)

```python
from radiologist.etl import filter_iqr, filter_lung_out_of_frame

df = filter_iqr(df, columns=["haralick_mean"], factor=1.5)
df = filter_lung_out_of_frame(df)
```

Both functions modify `excluded` and `exclusion_reason` in place on the loaded DataFrame and return it. They do not drop rows; exclusion is a flag so the manifest remains complete.

### Deterministic splitting (`radiologist.etl.split`)

```python
from radiologist.etl import assign_split

split = assign_split("patient_001_ap.png", ratios={"train": 0.70, "val": 0.15, "test": 0.15})
# → "train"  (deterministic: same filename always yields same split)
```

The MD5 hex digest of the filename is converted to a fraction in `[0, 1)` and mapped to a cumulative-ratio bracket. No split state is stored.

### Shard construction (`radiologist.etl.shards`)

```python
from radiologist.etl import build_shards

updated_manifest_path = build_shards(
    manifest_path="gs://bucket/manifest-abc123.jsonl",
    shard_root="gs://bucket/shards/",
    ratios={"train": 0.70, "val": 0.15, "test": 0.15},
    shard_size=1000,
)
```

Writes `{shard_root}/{split}/{label}/{split}-{label}-{idx:06d}.tar` files. The manifest is rewritten in place with the `shard` field set on every included record.

## Running the pipeline

### Via Hydra (recommended)

```bash
cd radiologist-etl
uv run --active python -m radiologist.etl.prefect_pipelines \
    source=gs://radiologist-liora-gcs/raw_chest_x_ray_data/images \
    destination=gs://radiologist-liora-gcs/manifests/ \
    shard_root=gs://radiologist-liora-gcs/shards/
```

Override any config key on the command line. Common overrides:

| Key | Default | Description |
|---|---|---|
| `source` | GCS URI | Raw image directory |
| `masks_root` | GCS URI | Segmentation mask directory |
| `destination` | GCS URI | Manifest output directory |
| `iqr_factor` | `1.5` | IQR multiplier for outlier threshold |
| `split_ratios.train` | `0.70` | Train fraction |
| `workers` | `null` (all CPUs) | Parallel worker count |
| `build_shards` | `true` | Whether to produce tar shards |
| `shard_size` | `1000` | Max images per tar shard |
| `resume_from_manifest` | `false` | Skip stats/filter/split, go straight to sharding |

### Resume from a previous stage

```bash
uv run --active python -m radiologist.etl.prefect_pipelines \
    resume_from_manifest=true \
    destination=gs://bucket/manifests/ \
    build_shards=true
```

### Programmatic use

```python
from radiologist.etl import (
    StatsProcessor, make_haralick,
    filter_iqr, filter_lung_out_of_frame,
    assign_split, build_shards,
    JsonlWriter,
)
```

## Configuration reference

Full config lives at `src/radiologist/etl/conf/etl.yaml`. The entry point is `radiologist.etl.prefect_pipelines:main` decorated with `@hydra.main`.

## Dependencies

Core: `radiologist-utils`, `fsspec`, `gcsfs`, `numpy`, `Pillow`, `scikit-image`, `pandas`, `pyarrow`, `webdataset`.

Optional: `prefect` (orchestration, install via `--extra prefect`). When not installed, `@flow` and `@task` are identity decorators and the pipeline runs as plain Python.
