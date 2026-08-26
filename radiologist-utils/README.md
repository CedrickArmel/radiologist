# radiologist-utils

[![ci](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml/badge.svg)](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CedrickArmel/radiologist/branch/main/graph/badge.svg)](https://codecov.io/gh/CedrickArmel/radiologist)
[![PyPI](https://img.shields.io/pypi/v/radiologist-utils)](https://pypi.org/project/radiologist-utils/)
![tested on](https://img.shields.io/badge/tested%20on-ubuntu--latest%20%7C%20python%203.10-blue)

Shared foundation library for the Radiologist mono-repo. Every other package depends on it; it has no internal workspace dependencies.

## What it solves

Provides three capabilities needed across the entire pipeline:

1. **Filesystem-agnostic I/O** — reading images and joining paths works identically whether data lives on a local disk, Google Cloud Storage, or any other fsspec-compatible remote.
2. **Structured logging** — thin `logging.LoggerAdapter` wrappers for plain CLI scripts and for distributed Lightning training (rank-aware, rank-zero-only gating).
3. **ML training utilities** — reproducible seeding, Kaiming / Xavier weight initialisation, Hydra config helpers, and Lightning callback/logger instantiation from config.

## Public API

All symbols below are importable from `radiologist.utils` or `radiologist.utils.ml`.

### Filesystem helpers (`radiologist.utils`)

| Symbol | Purpose |
|---|---|
| `pathjoin(a, *paths) -> str` | Join path segments, preserving fsspec protocols (`gs://`, `s3://`, …) |
| `pathname(path) -> str` | Protocol-aware `Path.name` |
| `pathparent(path) -> str` | Protocol-aware `Path.parent` |
| `pathstem(path) -> str` | Protocol-aware `Path.stem` |
| `ImageReader(source, storage_options)` | Factory returning `LocalImageReader` or `RemoteImageReader` based on URI scheme |
| `read_image(source, storage_options)` | Read a single PNG/JPEG from any fsspec URI → `(np.ndarray, metadata)` |

`BaseImageReader` is an abstract lazy iterator over `(np.ndarray, dict)` tuples via `.iterate()`. Call it to stream images without loading the entire dataset into memory.

```python
from radiologist.utils import ImageReader

reader = ImageReader("gs://my-bucket/images/", storage_options={"token": "anon"})
for image, meta in reader.iterate():
    process(image, meta)
```

### Logging (`radiologist.utils`)

| Symbol | Purpose |
|---|---|
| `Logger(name, extra)` | Plain `LoggerAdapter` for CLI scripts |
| `RankedLogger(name, rank_zero_only, extra)` | Distributed-safe adapter; suppresses non-rank-0 output when `rank_zero_only=True` |

### ML utilities (`radiologist.utils.ml`)

| Symbol | Purpose |
|---|---|
| `set_seed(seed, ...)` | Seeds Python, NumPy, PyTorch, CUDA; sets `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG` |
| `seed_worker(worker_id)` | DataLoader `worker_init_fn` for reproducible multi-process loading |
| `get_seeded_generator(seed)` | Returns a seeded `torch.Generator` for DataLoader `generator=` |
| `initialize_weights(module, ...)` | Kaiming-init Conv layers, Xavier-init Linear layers |
| `instantiate_callbacks(callbacks_cfg)` | Build a list of Lightning callbacks from a Hydra DictConfig |
| `instantiate_loggers(logger_cfg)` | Build a list of Lightning loggers from a Hydra DictConfig |
| `sequential_scheduler(optimizer, schedulers, milestones)` | Compose multiple LR schedulers into a `SequentialLR` |
| `extras(cfg)` | Apply optional Hydra run-time extras (warning suppression, tag enforcement, config printing) |
| `task_wrapper(task_func)` | Decorator ensuring `wandb.finish()` is always called on exit |
| `get_metric_value(metric_dict, metric_name)` | Safely retrieve a scalar metric from `trainer.callback_metrics` |
| `log_hyperparameters(object_dict)` | Rank-zero: resolve full Hydra config and pass to every trainer logger |
| `print_config_tree(cfg, ...)` | Pretty-print a DictConfig as a Rich tree |
| `enforce_tags(cfg, ...)` | Prompt for W&B tags when absent; raise during multirun if tags are empty |

## Design notes

- `ImageReader` is a factory function (not a class) that returns a concrete subclass. All path helpers accept arbitrary fsspec URIs, making the whole stack remote-filesystem-compatible without any caller-side branching.
- Optional heavy dependencies (`torch`, `lightning`, `wandb`, `omegaconf`) are guarded with `try/except ImportError` stubs so the module imports cleanly without them.
- `RankedLogger` never imports Lightning at module level — it wraps the standard `logging` module only.

## Dependencies

Core: `fsspec`, `gcsfs`, `numpy`, `Pillow`, `rich`.

Optional (installed via extras): `omegaconf`, `hydra-core`, `torch`, `lightning`, `wandb`, `lightning_utilities`.
