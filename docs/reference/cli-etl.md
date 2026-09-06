# ETL CLI Reference

`radiologist etl` is the ETL command group of the unified `radiologist` CLI
(entry point `radiologist.cli.main:main`, package `radiologist-cli`),
fronting the three independent, Hydra-composed `radiologist-etl` pipeline
stages: [`extract`](#extract), [`assign-split`](#assign-split), and
[`build`](#build) — invoked as `radiologist etl <stage>`. It requires the
`etl` extra, which is what makes `radiologist.etl` importable at all:

```bash
pip install "radiologist-cli[etl]"
```

Unlike `infer`/`registry` (Typer-based, `--flag value` options), each stage
is a plain Hydra entry point
(`@hydra.main(config_path="pkg://radiologist.etl.conf", config_name=...)`),
so its parameters are Hydra config keys overridden with `key=value` syntax,
and every stage gets `--multirun` sweeps and `--config-name`/`--config-path`
overrides for free. `radiologist etl <stage> --help` prints that stage's
full composed config tree (Hydra's own generated help); `radiologist etl
--help` (no stage) lists the three stage names. See
[`config-etl.md`](config-etl.md) for background on the pipeline's
IQR-filtering and Haralick-GLCM parameters.

Every stage prints a single keyed record to stdout via
`radiologist.utils.cli.emit` — `kv` (default, `key=value` lines), `json`, or
`yaml`, selected with the global `--output`/`-o` flag or the
`RADIOLOGIST_OUTPUT` environment variable. The flag is parsed off the full
argv *before* the stage name is parsed, so it works before or after it:

```bash
radiologist --output json etl extract file_list=... destination=...
radiologist etl extract --output json file_list=... destination=...
```

## Input pre-flight check

Each stage calls an internal `_ensure_input_exists` check on its primary
input before doing any work, via `fsspec.url_to_fs` (honoring
`storage_options`), and fails fast with
`FileNotFoundError: {label} not found: {uri}` rather than running partway
into the flow and failing deep inside it:

| Stage | Input key checked | Label in the error message |
|---|---|---|
| `extract` | `file_list` | `File listing` |
| `assign-split` | `manifests_dir` | `Manifests folder` |
| `build` | `split_manifest` | `Split manifest` |

## Exit codes

Exceptions are mapped to a process exit code via
`radiologist.utils.cli.exit_code_for`: `2` for `FileNotFoundError` (this
includes the pre-flight check above), `1` for any other failure. A failing
run prints `Error: {message}` on stderr and emits no result record at all.

## `extract`

Extract Haralick GLCM stats for every image listed in `file_list`, filter
outliers by IQR, and write a JSONL extract manifest.

```bash
radiologist etl extract \
  file_list=gs://bucket/listings/images.txt \
  destination=gs://bucket/manifests/

# Override quality-filtering / Haralick params
radiologist etl extract \
  file_list=gs://bucket/listings/images.txt \
  destination=gs://bucket/manifests/ \
  iqr_factor=2.0 haralick.features='[mean,entropy]'
```

Emitted keys: `run_id`, `manifest_path`, `total`, `succeeded`, `failed`,
`failure_rate`, `excluded`.

| Key | Default | Description |
|---|---|---|
| `file_list` | `???` (required) | URI of the newline-delimited listing of images to process. |
| `destination` | `???` (required) | Folder that accumulates extract manifests. |
| `images_root` | `null` | Required when `masks_root` is set. |
| `masks_root` | `null` | Root folder for segmentation masks, mirrored path-for-path under `images_root`. |
| `iqr_columns` | `["haralick_mean"]` | Stat columns tested for IQR outliers. |
| `iqr_factor` | `1.5` | IQR multiplier defining the outlier fence. |
| `haralick.features` / `.distances` / `.angles` | `["mean"]` / `[1]` / `null` | Haralick GLCM feature-extraction params — see [`config-etl.md`](config-etl.md#haralick-glcm-features). |
| `workers` | `null` | Number of worker processes; `null` resolves to `default_workers()`. |
| `batch_size` | `64` | Batch size for stat extraction. |
| `max_failure_rate` | `0.0` | Tolerance for per-image extraction failures before the run fails. |
| `run_label` | `null` | Optional label folded into the run-ID hash, so an intentional re-run on unchanged data gets a new `run_id` instead of colliding. |
| `storage_options` | `null` | Extra kwargs forwarded to `fsspec.url_to_fs`. |
| `runner` | `local` | Execution backend group (`local`, `dask_local`, `dask_cluster`, `dask_address`, `ray_local`, `ray_cluster`, `beam_direct`, `beam_dataflow`); select with `runner=<name>`. Non-local backends need further `radiologist-etl` extras — see `radiologist-etl/README.md`. |

## `assign-split`

Deterministically assign every record across the manifests under
`manifests_dir` to a split (train/val/test) and write a split manifest.

```bash
radiologist etl assign-split \
  manifests_dir=gs://bucket/manifests/ \
  destination=gs://bucket/manifests/

# Override the split ratios (order matters; must sum to 1.0)
radiologist etl assign-split \
  manifests_dir=gs://bucket/manifests/ \
  destination=gs://bucket/manifests/ \
  split_ratios='[[train,0.8],[val,0.1],[test,0.1]]'
```

Emitted keys: `run_id`, `split_manifest_path`, `source_manifest_count`,
`record_count`, `duplicate_count`, `counts_by_split`.

| Key | Default | Description |
|---|---|---|
| `manifests_dir` | `???` (required) | Folder of `extract`-stage manifests to assign. |
| `destination` | `???` (required) | Folder the split manifest is written into. |
| `split_ratios` | `[["train",0.70],["val",0.15],["test",0.15]]` | Ordered `[name, fraction]` pairs — order is part of the split-stability contract, not a formatting detail. |
| `run_label` | `null` | Optional label folded into the run-ID hash. |
| `storage_options` | `null` | Extra kwargs forwarded to `fsspec.url_to_fs`. |

`counts_by_split` maps every configured split name to its record count and
always carries one entry per configured ratio, including splits that
received zero records. Being a nested mapping, it serialises as a nested
object under `--output json`/`--output yaml`, and flattens to one dotted
`counts_by_split.<split>=<count>` line per split under the default `kv`
format.

## `build`

Read the split manifest, shard the referenced images into WebDataset tar
shards, and write a build manifest plus a per-split report.

```bash
radiologist etl build \
  split_manifest=gs://bucket/manifests/split-abc123.jsonl \
  shard_root=gs://bucket/shards/

# Larger shards, explicit worker count
radiologist etl build \
  split_manifest=gs://bucket/manifests/split-abc123.jsonl \
  shard_root=gs://bucket/shards/ \
  shard_size=2000 workers=8
```

Emitted keys: `run_id`, `output_dir`, `manifest_path`, `report_path`,
`shard_count`, `record_count`, `failed`, `failure_rate`.

| Key | Default | Description |
|---|---|---|
| `split_manifest` | `???` (required) | Path to the `assign-split` stage's output manifest. |
| `shard_root` | `???` (required) | Directory where `{split}/{label}/{split}-{label}-{idx:06d}.tar` shards are written. |
| `shard_size` | `1000` | Max samples per tar shard. |
| `split_ratios` | `[["train",0.70],["val",0.15],["test",0.15]]` | Report-only echo of the configured ratios — never used for assignment at this stage. |
| `workers` | `null` | Number of worker processes; `null` resolves automatically. |
| `max_failure_rate` | `0.0` | Tolerance for unshardable records before the run fails. |
| `run_label` | `null` | Optional label folded into the run-ID hash. |
| `storage_options` | `null` | Extra kwargs forwarded to `fsspec.url_to_fs`. |
| `runner` | `local` | Execution backend group, same options as `extract`. |

## Python API equivalent

Every stage is a thin CLI wrapper over a `radiologist.etl` function taking a
composed Hydra `DictConfig`, so any workflow above can also be scripted
directly without the CLI:

```python
from hydra import compose, initialize_config_module

from radiologist.etl import run_extract

with initialize_config_module(config_module="radiologist.etl.conf", version_base=None):
    cfg = compose(
        config_name="extract",
        overrides=[
            "file_list=gs://bucket/listings/images.txt",
            "destination=gs://bucket/manifests/",
        ],
    )

result = run_extract(cfg)
```

`run_assign_split(cfg)` and `run_build(cfg)` follow the same pattern with
`config_name="assign_split"`/`"build"`. See the [API Reference](api-etl.md)
for their full return-type documentation (`ExtractResult`,
`AssignSplitResult`, `BuildResult`).
