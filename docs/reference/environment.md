# Environment Variables

The pages under **CLI & Config** document this project's own Hydra configs and CLI flags. Several
dependencies also read environment variables directly — outside Hydra, unaffected by any `key=value`
CLI override — before or during the calls this project makes into them. This page catalogs the ones
actually relevant given how each dependency is used here; it is not an exhaustive dump of each
library's env var surface.

## GCS / fsspec (`gcsfs`)

`radiologist-utils`, `radiologist-etl`, and `radiologist-core` route all remote I/O through
`fsspec.url_to_fs()` (see `radiologist-utils/src/radiologist/utils/filesystem.py`,
`readers.py`, and `radiologist-etl/src/radiologist/etl/prefect_pipelines.py`). Only the `gcsfs` extra
is installed anywhere in this repo (`gcs` extras in each package's `pyproject.toml`) — no `s3fs`/`adlfs`
— so GCS authentication is the only remote-filesystem env var surface that matters here.

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | unset | Path to a service-account JSON key. `gcsfs`'s default `google_default` auth method (`google.auth.default()`) checks this first, then falls back to gcloud's local ADC file, then the GCE metadata server, then anonymous access. |
| `GOOGLE_CLOUD_PROJECT` (legacy: `GCLOUD_PROJECT`) | unset | Fallback "quota project" for billing/quota when the credential itself doesn't carry one. |
| `NO_GCE_CHECK` | unset (`false`) | Set to `true` to skip the GCE metadata-server probe during auth — avoids a slow/hanging check when running outside GCE (e.g. locally or in CI). |

Per-call `storage_options` (see `docs/reference/config-etl.md`'s `storage_options` key) can override
credentials without touching the environment at all — prefer that for one-off overrides.

## Weights & Biases (`wandb`)

Used by `radiologist-core` (the `WandbLogger` training logger and ONNX/registry export),
`radiologist-registry` (`push`/`pull`/`promote`/`resolve`/`alias` CLI and library), and
`radiologist-inference` (`registry` extra — `from_registry`/`from_selector`).

| Variable | Default | Purpose |
|---|---|---|
| `WANDB_API_KEY` | unset | Auth token. Required non-interactively for anything that talks to W&B: training, the `radiologist-registry` CLI, and `radiologist`'s registry-backed commands (`--run-id`/`--tags`/etc., see [`cli-inference.md`](cli-inference.md)). |
| `WANDB_MODE` | `online` | `offline` logs locally without network access (sync later with `wandb sync`); `disabled` no-ops entirely — useful in CI/tests. |
| `WANDB_PROJECT` / `WANDB_ENTITY` | unset | Override the hardcoded `project`/`entity` in `radiologist-core/src/radiologist/core/configs/loggers/wandb.yaml` without editing the Hydra config. |
| `WANDB_BASE_URL` | `https://api.wandb.ai` | Point at a self-hosted/enterprise W&B server instead of the public SaaS. |
| `WANDB_DIR` / `WANDB_CACHE_DIR` / `WANDB_ARTIFACT_DIR` / `WANDB_DATA_DIR` | platform cache dir (e.g. `~/.cache/wandb`) | Redirect wandb's own local run files / artifact cache — distinct from Hydra's `${paths.output_dir}` (already passed as the logger's `save_dir`), so set these if you also want wandb's internal cache somewhere specific. |
| `WANDB_SILENT` / `WANDB_QUIET` | `false` | Suppress wandb's own console output. |
| `WANDB_TAGS` | unset | Comma-separated default run tags — an env-level alternative to the `tags=...` Hydra override (`extras.enforce_tags` fails the run if tags end up unset either way). |
| `WANDB_CORE_DEBUG` | `false` | See [Go runtime (`wandb-core`)](#go-runtime-wandb-core) below — this is the one wandb variable that's actually a Go-service verbosity switch, not a Python-side one. |

### Go runtime (`wandb-core`)

`wandb`'s installed package ships a compiled Go binary — `wandb/bin/wandb-core` (confirmed via its
embedded build info: `go1.26.4`) — which the Python SDK spawns as a subprocess (`service_process.py`)
to handle the actual run-streaming service; the subprocess inherits the full parent environment
(`subprocess.Popen(..., env=os.environ)`). This is the one place Go-level env vars are actually in
play in this project's dependency chain (no other dependency here is Go-based).

| Variable | Default | Purpose |
|---|---|---|
| `WANDB_CORE_DEBUG` | `false` | wandb-level switch: when truthy, the Python SDK launches `wandb-core` with `--log-level -4` (Go's `slog` debug level — `-4: debug, 0: info, 4: warn, 8: error`, confirmed via `wandb-core --help`). This is the supported way to get verbose logs out of the core service; there's no separate env var for intermediate levels. |
| `GODEBUG` | unset | Go runtime's own debug-settings string (e.g. `GODEBUG=http2debug=1,gctrace=1`) — inherited by the `wandb-core` subprocess like any other Go binary, since it reads directly from its process environment at startup. Independent of `WANDB_CORE_DEBUG`; only affects the Go runtime's own internals (HTTP/2 transport tracing, GC tracing, etc.), not wandb's own log messages. |
| `GOTRACEBACK` | `single` | Controls how much stack-trace detail Go dumps if `wandb-core` panics/crashes (`none`, `single`, `all`, `system`, `crash`) — useful when debugging a core-service crash rather than routine logging. |

## Prefect (`radiologist-etl`, optional `prefect` extra)

`radiologist.etl.prefect_pipelines` wraps every pipeline stage in `@flow`/`@task` (no-op stand-ins when
the `prefect` extra isn't installed — see the Gotchas section on optional extras).

| Variable | Default | Purpose |
|---|---|---|
| `PREFECT_API_URL` / `PREFECT_API_KEY` | unset | Point at a Prefect Cloud workspace or a self-hosted Prefect Server for orchestration/run visibility in its UI. Without these, Prefect falls back to its local ephemeral SQLite backend — pipelines still run identically, just without a server to observe them in. |
| `PREFECT_HOME` | `~/.prefect` | Local state/config directory (profiles, local server DB, logging config). |
| `PREFECT_LOGGING_LEVEL` | `INFO` | Verbosity of Prefect's own logger, independent of this project's own logging. |
| `PREFECT_LOCAL_STORAGE_PATH` | `~/.prefect/storage` | Where task results are persisted locally when persistence is enabled — relevant to the `cache_policy=INPUTS` set on every `@task` in `prefect_pipelines.py`, which keys caching off task inputs. |

## PyTorch / CUDA

| Variable | Default | Purpose |
|---|---|---|
| `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG` | set explicitly by `radiologist.utils.ml.set_seed()` (`seeding.py`) | Not something you set yourself in normal use — `set_seed()` sets both (`CUBLAS_WORKSPACE_CONFIG=":4096:8"`) for deterministic cuBLAS ops and hash-seed reproducibility, per the root README's Reproducibility section. `CUBLAS_WORKSPACE_CONFIG` only takes effect if set before any CUDA context exists, which is why `set_seed()` must run first. |
| `CUDA_VISIBLE_DEVICES` | all devices visible | Restrict which GPUs the process can see. Combines with `trainer.yaml`'s `devices: auto` — Lightning auto-detects among whatever this variable exposes. |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | Tune the CUDA caching allocator (e.g. `expandable_segments:True`) — useful if training hits fragmentation-related OOMs. |
| `OMP_NUM_THREADS` | platform default | CPU thread count for BLAS/OpenMP ops. Worth pinning explicitly alongside `datamodule/default.yaml`'s multi-worker `WebLoader`s to avoid oversubscription. |
| `TORCH_HOME` | `$XDG_CACHE_HOME/torch` (`~/.cache/torch`) | Where `torch.hub`/`torchvision` cache downloaded pretrained weights. Not currently exercised — `module/resnet50.yaml` builds `torchvision.models.resnet50` with no `weights=` argument (no download), and `trainable_layers: null` fully reinitializes anyway — but relevant if a custom module config opts into pretrained weights (see the "bring your own Hydra config" section of [`config-core.md`](config-core.md)). |

## Lightning

Only matters once a run goes beyond the default `strategy: auto` on a single node:

| Variable | Default | Purpose |
|---|---|---|
| `NODE_RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` | — | Standard multi-process/multi-node DDP coordination variables, read directly by Lightning's cluster environment plugins (`lightning.fabric.plugins.environments`) when manually launching multiple processes. |
| `SLURM_*` (`SLURM_NTASKS`, `SLURM_NODEID`, `SLURM_PROCID`, etc.) | — | Auto-detected by `SLURMEnvironment` when the job is submitted via `srun`/`sbatch` — no manual configuration needed in that case. |
| `PL_TORCH_DISTRIBUTED_BACKEND` | `nccl` (GPU) / `gloo` (CPU) | Override the distributed backend Lightning picks for DDP. |
