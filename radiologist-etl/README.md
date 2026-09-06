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

result = build_shards(
    split_manifest_path="gs://bucket/manifest-abc123.jsonl",
    shard_root="gs://bucket/shards/",
    # ORDERED pairs, never a mapping — see the split-stability contract above.
    # A mapping is rejected with ValueError rather than silently coerced.
    ratios=[("train", 0.70), ("val", 0.15), ("test", 0.15)],
    shard_size=1000,
)
result.run_id, result.manifest_path, result.shard_count, result.record_count
```

Writes `{shard_root}/{split}/{label}/{split}-{label}-{idx:06d}.tar` files and
returns a `BuildResult`. Here `ratios` is **report-only** — splits were already
assigned by the assign-split stage and are read off the manifest.

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

## Using the public API

The three stages are plain functions over plain arguments. Nothing in this
package requires Hydra, Prefect, or the CLI — those are layers *on top* of the
functions below, and the CLI does not do anything you cannot do here.

There are two distinct entry levels:

| Level | Symbols | Takes |
|---|---|---|
| Library | `extract`, `assign_splits`, `build_shards` | ordinary Python arguments |
| Hydra flow | `run_extract`, `run_assign_split`, `run_build`, `extract_flow`, `assign_split_flow`, `build_flow` | a composed `DictConfig` |

Use the library level for embedding; use the flow level only if you already have
a `DictConfig` and want the Prefect artifacts and runner resolution too.

### The three stages end to end

```python
from radiologist.etl import assign_splits, build_shards, extract

extracted = extract(
    file_list="gs://bucket/listings/images.txt",
    destination="gs://bucket/pipelines/manifests",
    images_root=None,
    masks_root=None,
    iqr_columns=["haralick_mean"],
    iqr_factor=1.5,
    workers=8,
    batch_size=64,
)

assigned = assign_splits(
    manifests_dir="gs://bucket/pipelines/manifests",
    destination="gs://bucket/pipelines/manifests",
    ratios=[("train", 0.70), ("val", 0.15), ("test", 0.15)],
)

built = build_shards(
    split_manifest_path=assigned.split_manifest_path,
    shard_root="gs://bucket/shards/",
    shard_size=1000,
)
```

Each stage returns a frozen result dataclass — `ExtractResult`
(`run_id`, `manifest_path`, `total`, `succeeded`, `failed`, `failure_rate`,
`excluded`), `AssignSplitResult` (`run_id`, `split_manifest_path`,
`source_manifest_count`, `record_count`, `duplicate_count`, `counts_by_split`),
`BuildResult` (`run_id`, `output_dir`, `manifest_path`, `report_path`,
`shard_count`, `record_count`, `failed`, `failure_rate`).

Every `run_id` is content-addressed: it hashes the stage's **input digest**
(the file listing's contents, the set of `extract-*.jsonl` files in the folder,
or the split manifest's contents) together with a digest of the
output-affecting config. Same input plus same config ⇒ same `run_id` and the
same output path. That is the mechanism that replaced the old
`resume_from_*` flags: re-running is idempotent by construction, and
`run_label` exists to force a *new* id when you deliberately want one.

### Individual helpers

```python
from radiologist.etl import (
    assign_split, filter_iqr, filter_lung_out_of_frame,
    JsonlWriter, ManifestRecord, ParquetWriter,
    plan_shards, process_batch, records_reader, write_shard,
)

# deterministic, stateless split assignment
assign_split("patient_001_ap.png", [("train", 0.70), ("val", 0.15), ("test", 0.15)])

# read a manifest back into ManifestRecord objects
records = records_reader("gs://bucket/pipelines/manifests/extract-<run_id>.jsonl")
```

The full exported surface is `radiologist.etl.__all__`.

## Extending via Hydra

This package has exactly one user-extensible config group: `runner/`. Everything
else in `extract.yaml` / `assign_split.yaml` / `build.yaml` is a flat value you
override with `key=value`.

### Adding an execution runner

The stage configs are packaged inside the wheel
(`config_path="pkg://radiologist.etl.conf"`), so you add group members through
an extra search path:

```
myconfigs/
└── runner/
    └── my_dask.yaml
```

```yaml
# myconfigs/runner/my_dask.yaml
# @package runner
family: dask
batch_size: 128
task_runner:
  _target_: prefect_dask.DaskTaskRunner
  cluster_class: dask_kubernetes.operator.KubeCluster
  cluster_kwargs:
    name: radiologist-etl
  adapt_kwargs:
    minimum: 2
    maximum: 40
```

```bash
radiologist etl extract \
    hydra.searchpath=[file:///abs/path/to/myconfigs] \
    runner=my_dask \
    file_list=gs://bucket/listings/images.txt \
    destination=gs://bucket/pipelines/manifests
```

`--config-dir /abs/path/to/myconfigs` is the equivalent Hydra flag.

The `# @package runner` header is required — every shipped `runner/*.yaml` has
it, and without it your keys land at `runner.runner` and the stage falls back to
the default plan.

The contract a runner config must satisfy:

| Key | Meaning |
|---|---|
| `family` | one of `local`, `dask`, `ray`, `beam`. This is what the availability check keys off — an unknown family is rejected, and `dask`/`ray`/`beam` report the extra to install when it is missing |
| `batch_size` | fallback dispatch/wave size, used by stages that have no top-level `batch_size` of their own (i.e. `build`) |
| `task_runner` | for `local`/`dask`/`ray`: a `_target_` for a Prefect `TaskRunner`, instantiated by Hydra |
| `beam` | for `family: beam` only: a `_target_` for `radiologist.etl.beam_executor.BeamExecutor` (Beam owns its own parallelism, so it is not a Prefect task runner) |

Because `pipeline_options` is handed to Beam verbatim, supporting a further Beam
runner (Flink, Spark, …) really is *just* a new `runner/*.yaml` with a different
`pipeline_options` mapping — no code change.

`assign-split` always runs locally and ignores `runner=` entirely.

### Custom feature extractors — use the Python API

`extract` accepts an `extractors` list of `StatExtractor` callables, but the
CLI/Hydra path does **not** expose that list as a config group today: the
extract flow builds `[make_haralick(...), lung_asymmetry]` in code, and only the
Haralick knobs (`haralick.features`, `haralick.distances`, `haralick.angles`)
are reachable as overrides. There is no `radiologist etl extract extractors=...`.

To run your own extractor, drive the stage through the public API — this is a
first-class supported path, not a workaround:

```python
# my_package/extractors.py  — must be importable at module level (see note below)
import numpy as np


def mean_intensity(image, metadata, mask=None):
    """Any callable matching StatExtractor: (image, metadata, mask=None) -> dict[str, float]."""
    return {"mean_intensity": float(np.asarray(image).mean())}
```

```python
from radiologist.etl import extract, make_haralick
from my_package.extractors import mean_intensity

result = extract(
    file_list="gs://bucket/listings/images.txt",
    destination="gs://bucket/pipelines/manifests",
    extractors=[make_haralick(features=["contrast"]), mean_intensity],
    iqr_columns=["mean_intensity"],   # your feature can drive the IQR filter
    workers=8,
    batch_size=64,
)
```

The `StatExtractor` protocol (`radiologist.etl.stats`):

```python
def __call__(
    self,
    image: np.ndarray,                 # H x W or H x W x C
    metadata: dict[str, str],
    mask: np.ndarray | None = None,    # None when no mask is available
) -> dict[str, float]: ...
```

It is a structural `Protocol` — a plain function or a `functools.partial`
qualifies, no inheritance and no registration.

Three constraints that are easy to miss:

- **Your extractor must be picklable.** The default mapper is a local process
  pool, so a module-level function or a `functools.partial` over one works; a
  lambda or a closure does not. This is exactly why `make_haralick` returns a
  `partial` rather than a nested function.
- **Returned keys become manifest columns**, so they are what you name in
  `iqr_columns`. Return `{}` when a required input (e.g. `mask`) is absent —
  that is what `lung_asymmetry` does.
- **Your extractor changes the `run_id`.** The extract run id folds in each
  extractor's identity (module-qualified name plus its partial args/kwargs), so
  swapping extractors produces a new content-addressed run rather than silently
  overwriting an existing manifest.

If you want the custom extractor *and* the Hydra/Prefect wrapper, the idiomatic
combination today is your own thin `@hydra.main` script that composes your own
config and calls `extract(...)` with `hydra.utils.instantiate` on an
extractors list you define. That is your script, not this package's config tree.

> **Known gap.** Making `extractors` a real Hydra config group
> (`conf/extractors/*.yaml` with `_target_` entries, instantiated in
> `extract_flow` instead of the hardcoded list) would be a small, contained
> change and would remove the need for the workaround above. It is not
> implemented — this section deliberately documents only what exists (see
> [#248](https://github.com/CedrickArmel/radiologist/issues/248)).

## Configuration reference

Each stage has its own Hydra config root under `src/radiologist/etl/conf/`:
`extract.yaml`, `assign_split.yaml`, `build.yaml` (plus `runner/*.yaml` for
the execution backends above). The CLI entry points live in
`radiologist-cli`, at `radiologist.cli.groups.etl`.

See [docs/reference/config-etl.md](../docs/reference/config-etl.md) for the
full per-key reference for all three stages.

## Dependencies

Core: `radiologist-utils`, `fsspec`, `numpy`, `scikit-image`, `pandas`, `pyarrow`, `webdataset`, `hydra-core`, `omegaconf`, `rich`.

Optional: `gcsfs` (GCS filesystem support, install via `--extra gcs`). `prefect` (orchestration, install via `--extra prefect`). When not installed, `@flow` and `@task` are identity decorators and the pipeline runs as plain Python. The default `local` runner requires no extra: without prefect it resolves to an execution plan carrying no task runner, so all three stages run on a plain, no-extras install. `dask`/`ray`/`beam` extras add the corresponding execution runner backend; only those three families can be reported unavailable, and the error names the extra to install.
