# ETL (Hydra) Configuration Reference

The ETL pipeline is **three independent stages**, not one flow. Each is its own
`radiologist etl` subcommand with its own Hydra config root, its own defaults,
and its own content-addressed run id:

| Stage | Command | Config | Entry point |
|---|---|---|---|
| Extract | `radiologist etl extract` | `extract.yaml` | `radiologist.cli.groups.etl:extract_main` |
| Assign split | `radiologist etl assign-split` | `assign_split.yaml` | `radiologist.cli.groups.etl:assign_split_main` |
| Build | `radiologist etl build` | `build.yaml` | `radiologist.cli.groups.etl:build_main` |

Each is decorated as:

```python
@hydra.main(
    config_path="pkg://radiologist.etl.conf",
    config_name="extract",      # or "assign_split" / "build"
    version_base=None,
)
```

The configs live inside the installed wheel at
`radiologist-etl/src/radiologist/etl/conf/`. The command bodies live in
`radiologist-cli`; `radiologist-etl` ships no CLI code and no console script.

These subcommands have no interactive `--help` for their parameters — every
parameter is a standard Hydra `key=value` override. `???` marks a mandatory key
with no default; the run fails at compose time if you do not supply it.

```bash
radiologist etl extract \
    file_list=gs://bucket/listings/images.txt \
    destination=gs://bucket/pipelines/manifests

radiologist etl assign-split \
    manifests_dir=gs://bucket/pipelines/manifests \
    destination=gs://bucket/pipelines/manifests

radiologist etl build \
    split_manifest=gs://bucket/pipelines/manifests/manifest-<run_id>.jsonl \
    shard_root=gs://bucket/shards
```

Add the global `--output json` flag (before or after the subcommand) for a
machine-readable result record.

Use `--cfg job` on any stage to print the fully composed config and exit without
running anything.

## Stage 1 — `extract`

Reads a newline-delimited listing of image URIs, computes per-image features,
applies quality filters, and writes one `extract-{run_id}.jsonl` manifest.

| Key | Default | Description |
|---|---|---|
| `file_list` | `???` | URI of the newline-delimited listing of images to process. Its **content digest** is half of the run id. |
| `destination` | `???` | Folder that accumulates extract manifests. |
| `images_root` | `null` | Root the listing's paths are relative to. Required when `masks_root` is set. |
| `masks_root` | `null` | Segmentation-mask directory, mirrored path-for-path under `images_root`. Setting it without `images_root` is rejected with a `ValueError` naming both, before any image is read. |
| `iqr_columns` | `["haralick_mean"]` | Feature columns tested for IQR outliers. `[]` disables IQR filtering. |
| `iqr_factor` | `1.5` | IQR multiplier defining the fence (`Q1 - factor*IQR`, `Q3 + factor*IQR`). |
| `haralick.features` | `["mean"]` | Subset of the nine Haralick properties (`mean`, `std`, `entropy`, `contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`, `ASM`). `null` computes all nine. An unknown name raises `ValueError`. |
| `haralick.distances` | `[1]` | Pixel-pair distances for the GLCM. `null` defaults to `[1]`. |
| `haralick.angles` | `null` | Angles in radians. `null` defaults to `[0, π/4, π/2, 3π/4]`. |
| `workers` | `null` | Worker count; `null` resolves to `radiologist.etl.default_workers()`. |
| `batch_size` | `64` | Images per dispatched batch. Overrides `runner.batch_size` for this stage. |
| `max_failure_rate` | `0.0` | Unreadable-image tolerance before the run fails. |
| `run_label` | `null` | Optional label folded into the run-id hash, so a deliberate re-run on unchanged inputs gets a fresh `run_id` instead of resolving to the existing one. |
| `storage_options` | `null` | Extra kwargs forwarded to `fsspec` for every remote read/write. |
| `runner` | `local` | Execution backend group — see [Execution runners](#execution-runners). |

Emitted record: `run_id`, `manifest_path`, `total`, `succeeded`, `failed`,
`failure_rate`, `excluded`.

### Feature extractors are not configurable here

`haralick.*` tunes the Haralick extractor's parameters. The **list** of
extractors is not a config group: the extract flow builds
`[make_haralick(...), lung_asymmetry]` in code, and there is no
`extractors=...` override. To run a custom extractor, drive `extract()` through
the Python API — see
[radiologist-etl/README.md](../pkg/etl.md) → *Extending via Hydra → Custom
feature extractors*.

## Stage 2 — `assign-split`

Merges every extract manifest in a folder, deduplicates, and assigns each
surviving record a split. Always runs locally — it has no `runner` group.

| Key | Default | Description |
|---|---|---|
| `manifests_dir` | `???` | Folder of extract manifests to merge. Only files starting with `extract-` and ending in `.jsonl` are selected, so a split manifest written by a previous run into the same folder is ignored and `destination` may equal `manifests_dir`. |
| `destination` | `???` | Where the merged split manifest is written. |
| `split_ratios` | `[["train", 0.70], ["val", 0.15], ["test", 0.15]]` | **Ordered sequence of `[name, fraction]` pairs.** |
| `run_label` | `null` | As above. |
| `storage_options` | `null` | Extra kwargs forwarded to `fsspec`. |

Emitted record: `run_id`, `split_manifest_path`, `source_manifest_count`,
`record_count`, `duplicate_count`, `counts_by_split`.

### `split_ratios` is an ordered sequence, not a mapping

The bracket order is part of the split contract. A filename's split is a pure
function of the filename and this ordered sequence alone, so a file's
assignment cannot flip as the corpus grows across incremental runs.
`normalize_ratios` **rejects a plain mapping outright with `ValueError`**
rather than coercing it (e.g. by sorting keys), because coercion would restore
the hidden order-dependence the contract removes.

Changing the shipped default order or fractions re-partitions every
already-processed corpus and must be treated as a breaking data change.

Duplicates are resolved first-occurrence-wins by sorted manifest name; the
count is reported in `duplicate_count`.

## Stage 3 — `build`

Turns a split manifest into WebDataset tar shards.

| Key | Default | Description |
|---|---|---|
| `split_manifest` | `???` | Manifest produced by `assign-split`. Its content digest is half of the run id. |
| `shard_root` | `???` | Root under which shards are written, at `{shard_root}/{run_id}/{split}/{label}/{split}-{label}-{idx:06d}.tar`. |
| `shard_size` | `1000` | Max records per tar shard. |
| `split_ratios` | `[["train", 0.70], ["val", 0.15], ["test", 0.15]]` | **Report only** — never used for assignment here. Splits were fixed by `assign-split` and are read off the manifest. |
| `workers` | `null` | Worker count; `null` resolves to `default_workers()`. |
| `max_failure_rate` | `0.0` | Unshardable-record tolerance before the run fails. |
| `run_label` | `null` | As above. |
| `storage_options` | `null` | Extra kwargs forwarded to `fsspec`. |
| `runner` | `local` | Execution backend group. This stage has no top-level `batch_size`, so it falls back to `runner.batch_size`. |

Emitted record: `run_id`, `output_dir`, `manifest_path`, `report_path`,
`shard_count`, `record_count`, `failed`, `failure_rate`.

## Run ids and re-running

There are no `resume_from_*` flags. That mechanism was removed in favour of
content-addressed run ids, which make re-running idempotent by construction.

Each stage derives a 16-character `run_id` by hashing its **input digest**
together with a digest of its output-affecting configuration:

| Stage | Input digest |
|---|---|
| `extract` | content digest of the `file_list` |
| `assign-split` | digest over the `extract-`-prefixed manifests in `manifests_dir` |
| `build` | content digest of the `split_manifest` |

Consequences worth internalising:

- Same inputs + same config ⇒ same `run_id` ⇒ same output path. Re-running is
  a no-op you can safely repeat.
- Any output-affecting change — a different `iqr_factor`, a different Haralick
  feature set, a different extractor — yields a different `run_id` and a
  separate manifest. Nothing is silently overwritten.
- To force a fresh id on genuinely unchanged inputs, set `run_label`.
- Because ids are derived rather than tracked, there is no run registry to
  consult: you resume by passing a previous stage's `manifest_path` /
  `split_manifest_path` into the next stage.

## Execution runners

`extract` and `build` accept `runner=<name>`, selecting the backend their
batches/shards are dispatched through. `assign-split` ignores it.

| Family | Configs | Extra | In `all`? |
|---|---|---|---|
| `local` (default) | `runner=local` | none | — |
| Dask | `runner=dask_local` / `dask_cluster` / `dask_address` | `radiologist-etl[dask]` | Yes |
| Ray | `runner=ray_local` / `ray_cluster` | `radiologist-etl[ray]` | No — opt-in only |
| Beam | `runner=beam_direct` / `beam_dataflow` | `radiologist-etl[beam]` | Yes |

Every `runner/*.yaml` carries `# @package runner` and defines:

| Key | Meaning |
|---|---|
| `family` | `local` \| `dask` \| `ray` \| `beam`. Drives the availability check; a missing extra is reported by name. |
| `batch_size` | Fallback dispatch/wave size for stages with no `batch_size` of their own (i.e. `build`). |
| `task_runner` | `_target_` for a Prefect `TaskRunner` (`local`, `dask`, `ray`). |
| `beam` | `_target_` for `radiologist.etl.beam_executor.BeamExecutor` — `family: beam` only. |

Configs with `???` inside them must be completed on the command line:
`dask_address` needs `runner.task_runner.address`, `dask_cluster` needs
`runner.task_runner.cluster_class`, `ray_cluster` needs
`runner.task_runner.address`, and both Beam configs need
`runner.beam.parts_dir` (plus GCP project/region/locations for
`beam_dataflow`).

Beam is the one family that is not a Prefect task runner: a Beam pipeline owns
its own parallelism and runner, so a Beam-backed stage is a single opaque unit
of work to the orchestrator. `runner.beam.parts_dir` is a scratch prefix the
pipeline writes per-unit outcomes to and reads back, because a Beam pipeline
cannot return a collection to its driver; for `beam_dataflow` it must be a
shared remote URI reachable by the workers.

```bash
radiologist etl extract \
    file_list=gs://bucket/listings/images.txt \
    destination=gs://bucket/pipelines/manifests \
    runner=beam_direct \
    runner.beam.parts_dir=/tmp/beam-parts
```

Prefect itself is optional. Without it, `@flow`/`@task` are identity decorators
and all three stages run as plain Python; the default `local` runner then
resolves to a plan carrying no task runner, so a no-extras install still runs
the whole pipeline.

## Adding your own runner

`runner/` is the one extensible config group in this package — every member
must carry the same `family` / `batch_size` / `task_runner` / `beam` shape
documented in [Execution runners](#execution-runners) above, plus the
`# @package runner` header (omitting it lands your keys at `runner.runner`
and the stage silently falls back to the default plan).

For the full worked example (`hydra.searchpath` vs. `--config-dir`, a sample
`myconfigs/runner/my_dask.yaml`, and the Beam-family variant) see
[radiologist-etl/README.md § Adding an execution runner](../pkg/etl.md#adding-an-execution-runner),
which also covers the pipeline-stage narrative, the manifest schema, and
programmatic usage.
