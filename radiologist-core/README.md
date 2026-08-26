# radiologist-core

[![ci](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml/badge.svg)](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CedrickArmel/radiologist/branch/main/graph/badge.svg)](https://codecov.io/gh/CedrickArmel/radiologist)
[![PyPI](https://img.shields.io/pypi/v/radiologist-core)](https://pypi.org/project/radiologist-core/)
![tested on](https://img.shields.io/badge/tested%20on-ubuntu--latest%20%7C%20python%203.10-blue)

The ML engine. Provides the Lightning training loop, streaming data module, focal loss, GradCAM attribution, and W&B model registry integration for chest X-ray classification.

## Business context

Once clean, labelled WebDataset shards exist (produced by `radiologist-etl`), this package trains a classifier capable of distinguishing:

- **Healthy** lung (no finding)
- **Viral** pneumonia / COVID (infectious infiltrates)
- **Opacity** (other lung opacities)

The trained model is exported as two ONNX artefacts — one for deterministic inference with visual explanations (GradCAM), one for uncertainty-aware inference (MC-Dropout) — and linked to a W&B Model Registry collection for downstream consumption by an inference or serving layer.

## Key capabilities

| Capability | What it enables |
|---|---|
| Class-balanced streaming | Trains on imbalanced clinical data without manual oversampling scripts |
| Prior-calibrated bias init | Logit outputs are calibrated to class prevalence from the first step |
| Focal Loss | Focuses learning on hard, misclassified examples rather than easy majority samples |
| GradCAM attribution | Produces heatmaps showing which lung region drove the prediction |
| MC-Dropout ONNX export | Enables uncertainty estimation at inference time (multiple stochastic passes) |
| HPO-friendly best-metric tracking | Optuna / W&B Sweeps receive the best epoch score, not the last |

## Package layout

```
radiologist-core/src/radiologist/core/
├── train.py            # Hydra entry point and train() function
├── module.py           # LModule (LightningModule)
├── losses.py           # FocalLoss
├── data/
│   ├── datamodule.py   # WebDatasetDataModule
│   └── shards.py       # shard discovery helpers
├── callbacks/
│   ├── attribution.py  # GradCAM + Integrated Gradients
│   ├── best_metric.py  # best-epoch metric tracking
│   └── wandb_summary.py
├── registry/
│   └── export.py       # export_onnx (checkpoint → deterministic + MC-Dropout ONNX)
└── configs/            # Hydra config tree
```

## Training a model

### Prerequisites

- Shards on GCS (or local) produced by `radiologist-etl`
- W&B API key in the environment (`WANDB_API_KEY`)

### Quick start

```bash
cd radiologist-core
uv run --active python -m radiologist.core.train
```

The defaults load from `src/radiologist/core/configs/train.yaml`. Override on the command line:

```bash
uv run --active python -m radiologist.core.train \
    trainer.max_epochs=30 \
    datamodule.shard_root=gs://my-bucket/shards/ \
    datamodule.split_manifest_uri=gs://my-bucket/manifests/manifest-abc123.jsonl \
    seed=42
```

### Evaluation only

```bash
uv run --active python -m radiologist.core.train \
    --config-name eval \
    ckpt_path=/path/to/checkpoint.ckpt
```

## Core components

### `LModule`

A backbone-agnostic `LightningModule`. Pass any `nn.Module` as `net`.

```python
from radiologist.core import LModule
from torchvision.models import resnet50

module = LModule(
    net=resnet50(num_classes=3),
    loss=FocalLoss(gamma=2, alpha=1),
    metric=MulticlassFBetaScore(num_classes=3, beta=1.0, average="macro"),
    optimizer=partial(AdamW, lr=1e-3, weight_decay=1e-2),
    scheduler=partial(sequential_scheduler, ...),
    trainable_layers=None,   # None = full re-init; list of dot-paths = fine-tune
    priors=None,             # overridden from datamodule at setup time
)
```

On `setup('fit')`, `LModule` does one of two things depending on `trainable_layers`:

- **`None`** — reinitialise all weights (Kaiming normal on Conv, Xavier on Linear). Use when training from scratch.
- **`["layer4", "fc"]`** — freeze all parameters, then selectively unfreeze by dot-path. Use when fine-tuning a pretrained backbone.

In both cases, if class priors are available (from the datamodule), the final `nn.Linear` bias is initialised to `−log(priors)`, giving calibrated starting logits.

### `WebDatasetDataModule`

Streams images from WebDataset tar shards. Handles class-balanced sampling automatically.

```python
from radiologist.core import WebDatasetDataModule

dm = WebDatasetDataModule(
    shard_root="gs://bucket/shards/",
    split_manifest_uri="gs://bucket/manifests/manifest-abc123.jsonl",
    label_map={"normal": "healthy", "pneumonia": "viral", "COVID": "viral"},
    batch_size=32,
)
```

The `label_map` collapses raw ETL folder names (e.g. `normal`, `COVID`) into model class names (e.g. `healthy`, `viral`). This decouples the dataset's folder structure from the model's output space.

Training dataloader: one `wds.WebDataset` pipeline per class with `resampled=True` (infinite streaming), combined via `wds.RandomMix` weighted by inverse class frequency, then unbatch → shuffle → rebatch for global shuffling within an epoch. Validation and test dataloaders are flat sequential pipelines with no resampling.

### `FocalLoss`

```python
from radiologist.core import FocalLoss

loss = FocalLoss(gamma=2.0, alpha=1.0, use_softmax=True, reduction="mean")
```

Applies softmax to logits, computes `alpha * (1 − pt)^gamma * −log(pt)`. Supports optional one-hot conversion and `mean | sum | none` reductions.

### Callbacks

#### `AttributionCallback`

Computes Layer GradCAM and Integrated Gradients every `every_n_val_epochs` validation epochs and on all test batches. Saves normalised PNGs to `{log_dir}/attributions/` and logs them to W&B. Skipped gracefully when `captum` is not installed.

```yaml
# configs/callbacks/default.yaml
attribution:
  _target_: radiologist.core.callbacks.AttributionCallback
  target_layer: layer4.1.conv2
  every_n_val_epochs: 5
  n_test_batches: 4
  n_samples_per_batch: 4
```

#### `BestMetricCallback`

Writes `best_{monitor}` to `trainer.callback_metrics` after every validation epoch. When training finishes, HPO frameworks (Optuna, W&B Sweeps) read the best epoch score rather than the last.

#### `WandbDefineSummaryCallback`

Calls `wandb.run.define_metric` at fit start so the W&B run summary panel highlights the best validation score automatically.

## Exporting and promoting a trained model

`radiologist.core.registry.export_onnx` turns a Lightning checkpoint into two local ONNX files — it has no W&B interaction:

```python
from radiologist.core.registry import export_onnx

result = export_onnx(
    ckpt_path="/path/to/checkpoint.ckpt",
    run_id="wandb-run-id",
    input_shape=(1, 1, 224, 224),
    classes=["healthy", "viral", "opacity"],
    cam_target_layer="layer4.1.conv2",
    out_dir="/tmp/onnx-export",
)
```

This produces:

1. **Deterministic** — `_CamWrapper` forward hook returns `(logits, activation)`. Useful for inference with visual explanation.
2. **MC-Dropout** — `nn.Dropout` layers left in training mode (`TrainingMode.PRESERVE`, no constant folding). Run multiple forward passes and aggregate for uncertainty estimation.

Uploading the exported ONNX files as W&B artifacts and linking them into a registry collection is handled by `radiologist-registry` (`WandbRegistry.log_model_artifacts()` then `WandbRegistry.promote()`, or the `radiologist-registry push` / `promote` CLI). See [`radiologist-registry/README.md`](../radiologist-registry/README.md) for the full flow.

## Configuration reference

The Hydra config tree lives at `src/radiologist/core/configs/`. Key files:

| File | Purpose |
|---|---|
| `train.yaml` | Root config; wires all sub-configs |
| `eval.yaml` | Evaluation-only mode (`train: false`, `ckpt_path: ???`) |
| `trainer.yaml` | Lightning Trainer (precision, gradient clipping, deterministic) |
| `datamodule/default.yaml` | Shard URIs, label map, transforms, normalisation |
| `module/resnet50.yaml` | ResNet-50 backbone, `num_classes` interpolated from datamodule |
| `module/loss/focal_loss.yaml` | γ=2, α=1, softmax, mean |
| `module/optimizer/adamw.yaml` | AdamW, lr=1e-3, weight_decay=1e-2 |
| `module/scheduler/sequential.yaml` | Linear warmup 500 steps → cosine annealing 10 000 steps |
| `module/metric/fbeta_score.yaml` | Macro F1 (`MulticlassFBetaScore`, β=1) |
| `callbacks/default.yaml` | BestMetric, WandbSummary, Attribution, ModelCheckpoint, LRMonitor |

## Optional extras

Install the `onnx-export` extra for ONNX export and W&B registry features:

```bash
uv add --active "radiologist-core[onnx-export]"
```

Adds: `onnx`, `onnxruntime`, `onnxscript`.

## Dependencies

Core: `radiologist-utils`, `torch`, `torchvision`, `lightning`, `webdataset`, `torchmetrics`, `wandb`, `hydra-core`.

Optional (`onnx-export`): `onnx`, `onnxruntime`, `onnxscript`.

Optional (`attribution`): `captum`.
