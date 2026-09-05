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

Three independently invocable stages, each with its own `radiologist etl`
subcommand, its own Hydra configuration root, and its own content-addressed
run ID:

```
Raw images (GCS or local)
  └─ 1. extract        → {destination}/extract-{run_id}.jsonl
       (Haralick GLCM + lung asymmetry, IQR/lung-out-of-frame quality flags)
  └─ 2. assign-split    → {destination}/manifest-{run_id}.jsonl
       (reads every extract manifest in a folder, dedupes, assigns splits)
  └─ 3. build           → {shard_root}/{run_id}/{split}/{label}/{split}-{label}-{idx:06d}.tar
       (shards the split manifest into streaming-ready WebDataset tars)
```

Re-running a stage over unchanged inputs lands on the same run ID and the
same artifacts, so a per-stage `resume_from_*` mechanism is unnecessary --
each stage is its own resumable unit. `extract` and `build` accept a
`runner=` override (see [Execution runners](#execution-runners) below);
`assign-split` always runs locally.

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

`make_haralick` returns a `StatExtractor` (a `Protocol`-typed callable). Any function with signature `(image, metadata, mask?) -> dict[str, float]` can be plugged into `process_batch` in its place.

`lung_asymmetry(image, metadata, mask)` — returns `asymmetry_ratio` and `asymmetry_diff` from a lung segmentation mask.

### Quality filtering (`radiologist.etl.filters`)

```python
from radiologist.etl import filter_iqr, filter_lung_out_of_frame

df = filter_iqr(df, columns=["haralick_mean"], factor=1.5)
df = filter_lung_out_of_frame(df)
```

Both functions modify `excluded` and `exclusion_reason` in place on the loaded DataFrame and return it. They do not drop rows; exclusion is a flag so the manifest remains complete.

### Deterministic splitting (`radiologist.etl.split`, `radiologist.etl.assign`)

```python
from radiologist.etl import assign_split

split = assign_split("patient_001_ap.png", [("train", 0.70), ("val", 0.15), ("test", 0.15)])
# → "train"  (deterministic: same filename always yields same split)
```

The MD5 hex digest of the filename is converted to a fraction in `[0, 1)` and mapped to a cumulative-ratio bracket. No split state is stored.

**Guaranteed property — split stability.** Split ratios are an explicitly
ordered sequence of `(name, fraction)` pairs, never a mapping — the bracket
order is part of the split contract, not a formatting detail. A filename's
split is a pure function of the filename and this ordered sequence alone;
it never depends on which other filenames are present, so a file's
train/val/test assignment cannot flip as the corpus grows across
incremental `assign_splits` runs. `assign_split`/`normalize_ratios` reject a
plain mapping outright with `ValueError` rather than silently coercing it
(e.g. by sorting its keys), because coercion would restore the hidden
order-dependence this contract removes. **Changing the shipped default
order or fractions (`train`, `val`, `test` = `0.70`, `0.15`, `0.15`)
re-partitions every already-processed corpus and must be treated as a
breaking data change.**

The assign-split stage (`radiologist.etl.assign_splits`) reads every extract
manifest in a folder (sorted by name), deduplicates records by source path
(first occurrence — by sorted manifest name — wins; duplicates are counted
and logged), assigns each surviving record's split from its filename alone,
and writes one split manifest whose row shape matches today's manifest
exactly. This stage always runs locally — it accepts no runner
configuration.

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

Each stage is its own `radiologist etl` subcommand (implemented in
`radiologist-cli`; this package ships no CLI code of its own -- see
[Configuration reference](#configuration-reference)).

```bash
radiologist etl extract \
    file_list=gs://radiologist-liora-gcs/listings/images.txt \
    destination=gs://radiologist-liora-gcs/pipelines/manifests

radiologist etl assign-split \
    manifests_dir=gs://radiologist-liora-gcs/pipelines/manifests \
    destination=gs://radiologist-liora-gcs/pipelines/manifests

radiologist etl build \
    split_manifest=gs://radiologist-liora-gcs/pipelines/manifests/manifest-<run_id>.jsonl \
    shard_root=gs://radiologist-liora-gcs/shards
```

Each subcommand accepts arbitrary Hydra overrides on the command line.
Common ones:

| Subcommand | Key | Default | Description |
|---|---|---|---|
| `extract` | `file_list` | required | Newline-delimited listing of image URIs |
| `extract` | `masks_root` | `null` | Segmentation mask directory; requires `images_root`, which is used to mirror each image's relative path under it. Setting it without `images_root` is rejected immediately with a `ValueError` naming both, before any image is read. |
| `extract` | `iqr_factor` | `1.5` | IQR multiplier for outlier threshold |
| `extract` | `max_failure_rate` | `0.0` | Unreadable-image tolerance before the run fails |
| `extract` | `runner` | `local` | Execution backend (see below) |
| `assign-split` | `manifests_dir` | required | Folder of extract manifests to merge. Only files whose name starts with `extract-` and ends in `.jsonl` are selected — any other file, including a split manifest written by a previous run, is ignored, so `destination` may equal `manifests_dir` |
| `assign-split` | `split_ratios` | `[[train,.70],[val,.15],[test,.15]]` | Ordered split contract |
| `build` | `split_manifest` | required | Manifest produced by `assign-split` |
| `build` | `shard_size` | `1000` | Max images per tar shard |
| `build` | `max_failure_rate` | `0.0` | Unshardable-record tolerance before the run fails |
| `build` | `runner` | `local` | Execution backend (see below) |

### Execution runners

`extract` and `build` accept `runner=<family>`, selecting the Prefect task
runner the stage's batches/shards are dispatched through:

| Family | Config | Extra | In `all`? |
|---|---|---|---|
| `local` (default) | `runner=local` | none | — |
| Dask | `runner=dask_local` / `runner=dask_cluster` / `runner=dask_address` | `radiologist-etl[dask]` | Yes |
| Ray | `runner=ray_local` / `runner=ray_cluster` | `radiologist-etl[ray]` | No — opt-in only |
| Beam | `runner=beam_direct` / `runner=beam_dataflow` | `radiologist-etl[beam]` | Yes |

`assign-split` always runs locally and accepts no `runner=` override.

#### What `all` installs (and why Ray is not in it)

`radiologist-etl[all]` is a curated aggregate: it installs `gcs`, `prefect`,
`dask`, and `beam`, but **deliberately excludes `ray`**. The Ray execution
family (`runner=ray_local` / `runner=ray_cluster`) is still under
development (see #188) — Beam, by contrast, is *not* deferred: it shipped
(see #189) and is part of `all` like the other production-ready backends.

If you installed `radiologist-etl[all]`, selected `runner=ray_local` or
`runner=ray_cluster`, and hit an "install the ray extra" error, that is
expected, not a bug. Opt in explicitly:

```bash
# in-repo contributor, from a workspace checkout
uv sync --active --extra ray

# external consumer
pip install 'radiologist-etl[ray]'
```

The repo's own dev-setup (`make dev-install`) and both CI test jobs also
exclude the Ray extra, but through a **separate mechanism**: they install
with `--all-extras --no-extra ray`. `--all-extras` installs every extra
named in `pyproject.toml` directly, bypassing the `all` extra's own
composition entirely — so scoping `all` to leave Ray out is not, by itself,
enough to keep Ray out of those installs, and `--no-extra ray` is required
in addition. Removing `--no-extra ray` from the Makefile or CI workflows
would silently reintroduce Ray into those installs even though `all` still
excludes it.

The documentation build (`make docs-install` / `docs-build`) is the one
install path that still pulls in every extra, Ray included — `mkdocstrings`
needs every optional module importable to render its API reference, so that
target intentionally uses plain `--all-extras` with no exclusion.

#### Beam

Beam is the one family that is not a Prefect task runner: a Beam pipeline
owns its own parallelism and its own runner, so a Beam-backed stage is a
single opaque unit of work from the orchestrator's point of view. It is
configured through `runner.beam`:

| Key | Description |
|---|---|
| `runner.beam.parts_dir` | required; scratch prefix the pipeline writes its per-unit outcomes to, then reads back — a Beam pipeline cannot return a collection to its driver |
| `runner.beam.pipeline_options` | handed to Beam's `PipelineOptions` verbatim (runner name, project, region, temp/staging location, container image, …) |
| `runner.beam.storage_options` | extra kwargs forwarded to fsspec |

```bash
radiologist etl extract file_list=data/listing.txt \
  runner=beam_direct runner.beam.parts_dir=/tmp/beam-parts

radiologist etl build split_manifest=... runner=beam_dataflow \
  runner.beam.parts_dir=gs://bucket/beam-parts \
  runner.beam.pipeline_options.project=my-project \
  runner.beam.pipeline_options.region=europe-west1 \
  runner.beam.pipeline_options.temp_location=gs://bucket/tmp \
  runner.beam.pipeline_options.staging_location=gs://bucket/staging
```

`parts_dir` must be reachable by the Beam workers as well as by the driver:
a local path is fine for the direct runner, but pairing a non-direct runner
with a local `parts_dir` raises `ValueError` at construction. Supporting a
further Beam runner (Flink, Spark, …) is a new `conf/runner/*.yaml` with a
different `pipeline_options` mapping and no code change. Provisioning the
project, bucket, or cluster a non-direct runner needs is the operator's.

### Programmatic use

```python
from radiologist.etl import (
    make_haralick,
    filter_iqr, filter_lung_out_of_frame,
    assign_split, build_shards,
    run_extract, run_assign_split, run_build,
    JsonlWriter,
)
```

## Configuration reference

Each stage has its own Hydra config root under `src/radiologist/etl/conf/`:
`extract.yaml`, `assign_split.yaml`, `build.yaml` (plus `runner/*.yaml` for
the execution backends above). The CLI entry points live in
`radiologist-cli`, at `radiologist.cli.groups.etl`.

## Dependencies

Core: `radiologist-utils`, `fsspec`, `gcsfs`, `numpy`, `Pillow`, `scikit-image`, `pandas`, `pyarrow`, `webdataset`.

Optional: `prefect` (orchestration, install via `--extra prefect`). When not installed, `@flow` and `@task` are identity decorators and the pipeline runs as plain Python. The default `local` runner requires no extra: without prefect it resolves to an execution plan carrying no task runner, so all three stages run on a plain, no-extras install. `dask`/`ray`/`beam` extras add the corresponding execution runner backend; only those three families can be reported unavailable, and the error names the extra to install.
