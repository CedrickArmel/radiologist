# Radiologist

[![ci](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml/badge.svg)](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CedrickArmel/radiologist/branch/main/graph/badge.svg)](https://codecov.io/gh/CedrickArmel/radiologist)
[![PyPI](https://img.shields.io/pypi/v/radiologist)](https://pypi.org/project/radiologist/)
![tested on](https://img.shields.io/badge/tested%20on-ubuntu--latest%20%7C%20python%203.10-blue)

A fully reproducible machine-learning pipeline for chest X-ray classification. Takes a raw archive of labelled radiographs and produces a production-ready classifier distinguishing healthy lungs from viral pneumonia / COVID-19 and other opacities — complete with visual explanations and uncertainty estimates.

## Business value

| Capability | What it enables |
|---|---|
| Automated quality gating | Flags images where lungs fall outside the frame or whose texture statistics are outliers before any model trains on them |
| Deterministic, reproducible splits | Every image is assigned to train / val / test by a hash of its filename; reruns always produce the same cohort without storing split state |
| Class-balanced streaming training | Clinical datasets are heavily imbalanced; the pipeline compensates automatically so the model does not specialise on the majority class |
| GradCAM visual explanations | Every prediction can be accompanied by a heatmap highlighting the lung region that drove the decision — enabling radiologist review |
| Uncertainty estimation | MC-Dropout at inference time surfaces cases where the model is uncertain, flagging them for human review |
| W&B Model Registry integration | Versioned, auditable model promotion to a central registry; downstream services pull a specific alias (`production`, `staging`) |

## Architecture overview

```
                     Raw X-rays (GCS or local)
                              │
                              ▼
                   ┌────────────────────┐
                   │   radiologist-cli   │  Unified `radiologist` CLI — operator-facing
                   │  (etl/core/registry │  entry point fronting every command group
                   │       /infer)       │
                   └──────────┬──────────┘
                              │
      ┌───────────────────────┼───────────────────────┬─────────────────────────┐
      ▼                       ▼                       ▼                         ▼
┌───────────────┐   ┌───────────────────┐   ┌───────────────────┐   ┌─────────────────────┐
│ radiologist-  │   │  radiologist-core  │   │   radiologist-     │   │    radiologist-      │
│     etl       │──▶│  Lightning train   │──▶│    registry        │──▶│    inference          │
│ extract →     │   │  loop → ONNX       │   │  Promote/resolve   │   │  ONNX Runtime + FastAPI│
│ assign-split  │   │  export            │   │  W&B Registry      │   │  serving               │
│ → build       │   │                    │   │  artefacts         │   │                        │
└───────────────┘   └───────────────────┘   └───────────────────┘   └─────────────────────┘
```

The shared foundation (`radiologist-utils`) sits below all layers.
`radiologist-app` (UI) is planned but does not yet exist on disk.

## Repository layout

This is a UV workspace mono-repo. Each package is independently installable and testable.

| Package | Purpose |
|---|---|
| [`radiologist-utils`](radiologist-utils/README.md) | Filesystem-agnostic I/O, logging, ML training utilities |
| [`radiologist-etl`](radiologist-etl/README.md) | Data preparation — Haralick GLCM, IQR filtering, sharding |
| [`radiologist-core`](radiologist-core/README.md) | Model training, evaluation, attribution, registry promotion |
| [`radiologist-inference`](radiologist-inference/README.md) | ONNX inference & serving — pull models from W&B Registry, serve via ONNX Runtime, FastAPI HTTP server |
| [`radiologist-registry`](radiologist-registry/README.md) | W&B model registry — promote, resolve, download ONNX artifacts |
| [`radiologist-cli`](radiologist-cli/README.md) | Unified `radiologist` CLI — single dispatcher fronting the `etl`/`core`/`registry`/`infer` command groups |

## Tech stack

**Training:** PyTorch · Lightning · torchvision (ResNet-50) · TorchMetrics

**Data:** WebDataset · fsspec · GCS · DVC

**Experiment tracking:** Weights & Biases · Hydra

**ETL:** Prefect · scikit-image (Haralick GLCM) · Parquet · pyarrow

**Serving:** FastAPI · ONNX Runtime

**Tooling:** UV · PyEnv · pre-commit · tox · mypy · black · isort

## Quick start

### 1 — Set up the environment

```bash
pyenv activate radiologist
make dev-install
```

### 2 — Prepare the data

The ETL pipeline is three independent stages, each its own subcommand of the
unified CLI. Each writes a content-addressed run id you feed to the next:

```bash
pip install 'radiologist-cli[etl]'

radiologist etl extract \
    file_list=gs://my-bucket/listings/images.txt \
    destination=gs://my-bucket/manifests/

radiologist etl assign-split \
    manifests_dir=gs://my-bucket/manifests/ \
    destination=gs://my-bucket/manifests/

radiologist etl build \
    split_manifest=gs://my-bucket/manifests/manifest-<run_id>.jsonl \
    shard_root=gs://my-bucket/shards/
```

See [`docs/reference/config-etl.md`](docs/reference/config-etl.md) for every
key, the execution-runner options, and how run ids make re-runs idempotent.

### 3 — Train

```bash
radiologist core \
    datamodule.shard_root=gs://my-bucket/shards/ \
    datamodule.split_manifest_uri=gs://my-bucket/manifests/manifest-abc123.jsonl \
    tags=[baseline]
```

`tags` is effectively required: `extras.enforce_tags: true` fails the run fast
when it is unset, so W&B runs stay searchable. See
[`docs/reference/config-core.md`](docs/reference/config-core.md) for the full
config tree and how to plug in your own loss, metric, or backbone.

### 4 — Promote to the model registry

Training already logged both ONNX exports (deterministic + MC-Dropout) as W&B artifacts. Link them into a registry collection:

```bash
radiologist registry promote entity/project/model-artifact \
    --run-id wandb-run-id \
    --det-collection chest-xray-classifier \
    --mcd-collection chest-xray-classifier-mcd
```

This emits a keyed record (`det_qualified_name=... mcd_qualified_name=... alias=...`) on stdout — pass `--output json` for a single parseable JSON object.

This resolves the deterministic artifact by `run_id` and the MC-Dropout artifact by the `{run_id}-mcd` convention, then links both into their collections under the same alias — `production` if neither collection has one yet, `staging` otherwise.

Equivalent Python API:

```python
from radiologist.registry import WandbRegistry

result = WandbRegistry().promote(
    path="entity/project/model-artifact",
    run_id="wandb-run-id",
    det_collection="chest-xray-classifier",
    mcd_collection="chest-xray-classifier-mcd",
)
```

See [`docs/reference/cli-registry.md`](docs/reference/cli-registry.md) for the full `radiologist registry` CLI reference (`push`, `pull`, `resolve`, `list`, `alias`, `transition-to-production`).

### 5 — Run the test suite

```bash
make test           # all packages
make test-core      # radiologist-core only
```

## Key design decisions

**Remote-first filesystem.** All I/O goes through fsspec. Switching from local to GCS (or S3) requires only a URI change — no code changes anywhere.

**Deterministic ETL run IDs.** The ETL run ID is a SHA-256 digest of the config dict plus a dataset fingerprint (file count + total bytes + run label). Rerunning with the same config and same data always produces the same ID, making runs idempotent and traceable.

**Two ONNX exports from one checkpoint.** The registry step exports two artefacts: a deterministic model that returns `(logits, activation)` for GradCAM visualisation, and an MC-Dropout model with stochastic passes preserved for uncertainty estimation. Both are linked to the same registry entry so consumers choose the variant they need.

**Optional extras everywhere.** Heavy optional dependencies (`prefect`, `captum`, `onnx`, `wandb`) are wrapped in `try/except ImportError` stubs. The package imports cleanly without them; the feature gracefully no-ops or raises a clear install message when invoked without its extra.

**Prior-calibrated logit initialisation.** The final classification layer's bias is initialised to `−log(prior)` where `prior` comes from the training set class frequencies. This prevents the model spending the first epochs learning the class imbalance; it starts with calibrated logits instead.

## Reproducibility

- Python version pinned to 3.10 via `.python-version`
- `uv.lock` pins every transitive dependency
- `set_seed` seeds Python, NumPy, PyTorch, CUDA and sets `CUBLAS_WORKSPACE_CONFIG` and `PYTHONHASHSEED`
- `trainer.yaml` sets `deterministic: true`
- ETL split assignment is MD5-hash-based — independent of insertion order or random state

## License

See [LICENSE](LICENSE).
