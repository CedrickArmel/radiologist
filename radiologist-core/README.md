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
├── train.py            # train() orchestration function (the Hydra entry point lives in radiologist-cli)
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

This package ships no CLI of its own. Training is the `core` group of the
unified `radiologist` command, which lives in `radiologist-cli`:

```bash
pip install radiologist-cli    # the core group is a hard dependency, no extra needed
radiologist core
```

Defaults are composed from `src/radiologist/core/configs/train.yaml`. The group
takes arbitrary Hydra overrides — everything after `core` is forwarded to Hydra
verbatim:

```bash
radiologist core \
    trainer.max_epochs=30 \
    datamodule.shard_root=gs://my-bucket/shards/ \
    datamodule.split_manifest_uri=gs://my-bucket/manifests/manifest-abc123.jsonl \
    tags=[baseline] \
    seed=42
```

`tags` is not optional in practice: `extras.enforce_tags: true` fails the run
fast when it is unset, so W&B runs stay searchable.

On success the command emits a six-key record (`run_id`, `best_ckpt_path`,
`det_onnx_path`, `mcd_onnx_path`, `det_qualified_name`, `mcd_qualified_name`);
add the global `--output json` flag for machine-readable output.

### Evaluation only

`eval.yaml` is a second top-level config, selected with Hydra's `--config-name`
flag rather than a separate subcommand:

```bash
radiologist core --config-name eval ckpt_path=/path/to/checkpoint.ckpt
```

It sets `train: false`, `test: true`, `loggers: null`, and makes `ckpt_path`
mandatory (`???`).

## Core components

### `LModule`

A backbone-agnostic `LightningModule`. It takes a **single** `DictConfig` and
instantiates every collaborator itself through `hydra.utils.instantiate`, so the
same object is constructible from the packaged config tree or from a config you
build in Python:

```python
from omegaconf import OmegaConf
from radiologist.core import LModule

module_cfg = OmegaConf.create(
    {
        "net": {"_target_": "torchvision.models.resnet50", "num_classes": 3},
        "loss": {
            "_target_": "radiologist.core.FocalLoss",
            "gamma": 2, "alpha": 1, "to_onehot_y": True,
            "use_softmax": True, "reduction": "mean",
        },
        # metric/optimizer/scheduler are *partial* factories — `_partial_: true`
        "metric": {
            "_target_": "torchmetrics.classification.MulticlassFBetaScore",
            "_partial_": True, "num_classes": 3, "beta": 1.0, "average": "macro",
        },
        "optimizer": {"_target_": "torch.optim.AdamW", "_partial_": True, "lr": 1e-3},
        "scheduler": None,          # None is allowed: no LR schedule
        "trainable_layers": None,   # None = full re-init; dict of dot-paths = fine-tune
        "priors": None,             # None = taken from the datamodule at setup time
    }
)
lm = LModule(cfg=module_cfg)
```

The config contract `LModule` enforces:

| Key | Requirement |
|---|---|
| `net` | fully instantiated `nn.Module`; called as `net(x)` and expected to return logits |
| `loss` | fully instantiated callable; called as `loss(logits, target)` where `target` is a `torch.long` class-index tensor of shape `(B,)` |
| `metric` | **partial** factory (`_partial_: true`). Called with no arguments twice (val + test) and must return a `torchmetrics.Metric` exposing `compute_on_cpu`, `sync_on_compute`, and `process_group` — those three are copied onto the internal `MeanMetric` loss trackers |
| `optimizer` | **partial** factory; called as `optimizer(params=...)` |
| `scheduler` | **partial** factory or `None`; called as `scheduler(optimizer=...)`, stepped per **step**, not per epoch |
| `trainable_layers` | `None`, or a mapping of dot-path → `None` (unfreeze whole submodule) or list of ints (unfreeze those indices) |

Note the metric is updated with **hard predicted class indices**
(`logits.argmax(dim=1)`), not probabilities — probability-based metrics such as
`MulticlassAUROC` will not behave meaningfully here.

`cfg` also carries:

- `trainable_layers` — `None` for full re-init, or a mapping of dot-paths to parameter-index lists (e.g. `{"layer4": None, "fc": None}`) to freeze everything else and fine-tune those submodules.
- `priors` — optional class prior probabilities for bias initialisation; falls back to the datamodule's `priors` attribute when unset.

On `setup('fit')`, `LModule` does one of two things depending on `trainable_layers`:

- **`None`** — reinitialise all weights (Kaiming normal on Conv, Xavier on Linear). Use when training from scratch.
- **a mapping** — freeze all parameters, then selectively unfreeze the named dot-paths. Use when fine-tuning a pretrained backbone.

In both cases, if class priors are available (from `cfg` or the datamodule), the final `nn.Linear` bias is initialised to `−log(priors)`, giving calibrated starting logits.

### `WebDatasetDataModule`

Streams images from WebDataset tar shards. Handles class-balanced sampling
automatically. Eight arguments are required — the two transform pipelines and
the two `WebLoader` partial factories have no defaults, so the practical way to
build one outside the CLI is to instantiate it from the packaged config:

```python
from hydra import compose, initialize_config_module
from hydra.utils import instantiate

with initialize_config_module("radiologist.core.configs", version_base="1.3"):
    cfg = compose(
        config_name="train",
        overrides=[
            "datamodule.shard_root=gs://bucket/shards/",
            "datamodule.split_manifest_uri=gs://bucket/manifests/manifest-abc123.jsonl",
            "datamodule.batch_size=32",
        ],
    )

dm = instantiate(cfg.datamodule)
```

Each batch is a dict with keys `input` (image tensor), `target` (`torch.long`
class index), and `key` (the WebDataset sample key).

The `label_map` collapses raw ETL folder names (e.g. `normal`, `covid`) into
model class names (e.g. `healthy`, `viral`). This decouples the dataset's folder
structure from the model's output space.

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

Uploading the exported ONNX files as W&B artifacts and linking them into a registry collection is handled by `radiologist-registry` (`WandbRegistry.log_model_artifacts()` then `WandbRegistry.promote()`, or the `radiologist registry push` / `radiologist registry promote` commands). See [`radiologist-registry/README.md`](../radiologist-registry/README.md) for the full flow.

## Using the public API

Everything below is reachable from `radiologist.core`'s `__init__.py` — the
package's supported surface. Nothing here needs the CLI or a Hydra run
directory; use it when you are embedding training in a notebook, a sweep
driver, or another service.

```python
from radiologist.core import (
    AttributionCallback,
    BestMetricCallback,
    FocalLoss,
    LModule,
    OnnxExportCallback,
    train,
    WandbDefineSummaryCallback,
    WebDatasetDataModule,
)
```

`radiologist.core.registry` exports one further symbol, `export_onnx`, from its
own `__init__.py`.

### Driving a full run from Python

`train(cfg)` is the whole training/evaluation orchestration — seeding, callback
and logger instantiation, datamodule and module construction, `fit`, `test` —
and it is a plain function over a `DictConfig`. Compose the config yourself
instead of going through `@hydra.main`:

```python
from hydra import compose, initialize_config_module
from radiologist.core import train

with initialize_config_module("radiologist.core.configs", version_base="1.3"):
    cfg = compose(
        config_name="train",
        overrides=[
            "trainer.max_epochs=5",
            "tags=[notebook]",
            "datamodule.shard_root=gs://bucket/shards/",
        ],
    )

metric_dict, object_dict = train(cfg)
print(metric_dict["val_score"])
print(object_dict["trainer"].checkpoint_callback.best_model_path)
```

`train` returns `(metric_dict, object_dict)`: the first is the merged
train/test callback metrics, the second holds the live `cfg`, `datamodule`,
`module`, and `trainer` objects. `paths.output_dir` resolves from Hydra's
runtime output dir, so under `compose()` (no run directory) point Lightning
somewhere explicit with `trainer.default_root_dir=...` if you need artifacts on
disk.

### Composing the pieces by hand

The building blocks are independent — you can wire them into your own Lightning
`Trainer` without `train()` at all:

```python
import lightning as L
from radiologist.core import BestMetricCallback, LModule

lm = LModule(cfg=module_cfg)            # see the LModule contract above
trainer = L.Trainer(
    max_epochs=5,
    accelerator="cpu",
    callbacks=[BestMetricCallback(monitor="val_score", mode="max")],
)
trainer.fit(lm, datamodule=dm)
```

### Checkpoint → ONNX, without W&B

`export_onnx` is pure local file work; it performs no registry calls:

```python
from radiologist.core.registry import export_onnx

result = export_onnx(
    ckpt_path="/path/to/checkpoint.ckpt",
    run_id="wandb-run-id",
    input_shape=(1, 3, 224, 224),
    classes=["healthy", "viral", "opacity"],
    cam_target_layer="layer4.1.conv2",
    out_dir="/tmp/onnx-export",
    opset=18,
)
result.det_path, result.mcd_path
```

Uploading and promoting those files is `radiologist-registry`'s job — see
[Using the public API](../radiologist-registry/README.md#using-the-public-api)
there.

### Loss and callbacks standalone

`FocalLoss` is an ordinary `nn.Module` and the four callbacks are ordinary
Lightning callbacks; both are usable in any training loop:

```python
from radiologist.core import FocalLoss

loss = FocalLoss(gamma=2.0, alpha=1.0, use_softmax=True, reduction="mean")
```

## Configuration reference

The Hydra config tree lives at `src/radiologist/core/configs/`. Key files:

| File | Purpose |
|---|---|
| `train.yaml` | Root config; wires all sub-configs |
| `eval.yaml` | Evaluation-only mode (`train: false`, `ckpt_path: ???`) |
| `trainer.yaml` | Lightning Trainer (precision, gradient clipping, deterministic) |
| `datamodule/default.yaml` | Shard URIs, label map, transforms, normalisation |
| `module/resnet50.yaml` | ResNet-50 backbone; `num_classes: 3` is set here and interpolated *by* `module/metric/fbeta_score.yaml` |
| `module/loss/focal_loss.yaml` | γ=2, α=1, softmax, mean |
| `module/optimizer/adamw.yaml` | AdamW, lr=1e-3, weight_decay=1e-2 |
| `module/scheduler/sequential.yaml` | Linear warmup 500 steps → cosine annealing 10 000 steps |
| `module/metric/fbeta_score.yaml` | Macro F1 (`MulticlassFBetaScore`, β=1) |
| `callbacks/default.yaml` | BestMetric, WandbSummary, Attribution, ModelCheckpoint, LRMonitor, OnnxExport |

## Extending via Hydra

You do not need to fork or subclass anything to swap the loss, the metric, or
the whole backbone. Every one of those is a Hydra **config group** whose members
are selected by name and instantiated from a `_target_`, so your own class in
your own package is a first-class option.

The packaged configs live inside the installed wheel (`config_path` is
`pkg://radiologist.core.configs`), so you add options by pointing Hydra at an
extra search path. Two equivalent ways:

```bash
# as a config override (works anywhere, including inside a sweep)
radiologist core hydra.searchpath=[file:///abs/path/to/myconfigs] module/loss=my_loss

# as a Hydra CLI flag
radiologist core --config-dir /abs/path/to/myconfigs module/loss=my_loss
```

Your directory mirrors the packaged tree:

```
myconfigs/
├── module/
│   ├── my_net.yaml           # a whole backbone (top-level `module` group)
│   ├── loss/my_loss.yaml     # a loss   (`module/loss` group)
│   └── metric/my_metric.yaml # a metric (`module/metric` group)
```

> **The one thing that will silently bite you.** Every member of the `module`
> group and its sub-groups must start with a `# @package module` header, exactly
> as the shipped configs do. Hydra otherwise infers the package from the
> directory name and your `loss:` key lands at `module.loss.loss`, where
> `LModule` never looks — the run then fails with a confusing instantiation
> error instead of a config error. The shipped `callbacks/*.yaml` use
> `# @package callbacks`, and `datamodule/default.yaml` / `trainer.yaml` use
> `# @package _global_`; match whichever group you are extending.

### A custom loss

```yaml
# myconfigs/module/loss/my_loss.yaml
# @package module
loss:
  _target_: my_package.losses.LabelSmoothedCE
  smoothing: 0.05
```

```python
# my_package/losses.py
import torch
from torch import nn


class LabelSmoothedCE(nn.Module):
    def __init__(self, smoothing: float = 0.0) -> None:
        super().__init__()
        self._inner = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, num_classes) float; target: (B,) int64 class indices
        return self._inner(logits, target)
```

```bash
radiologist core hydra.searchpath=[file:///abs/path/to/myconfigs] module/loss=my_loss
```

The loss is instantiated **fully** (no `_partial_`) and called as
`loss(logits, target)`.

### A custom metric

```yaml
# myconfigs/module/metric/my_metric.yaml
# @package module
metric:
  _target_: torchmetrics.classification.MulticlassRecall
  _partial_: true
  num_classes: ${module.net.num_classes}
  average: macro
  compute_on_cpu: true
  sync_on_compute: true
```

The metric config **must** carry `_partial_: true`: `LModule` calls the result
twice with no arguments to build separate validation and test metric instances.
The returned object must be a `torchmetrics.Metric` — `LModule` reads
`compute_on_cpu`, `sync_on_compute` and `process_group` off it to configure the
loss trackers consistently.

It is updated as `metric(preds, target)` where `preds` are **hard class indices**
(`logits.argmax(dim=1)`), so choose a label-based metric. Probability-based
metrics (AUROC, calibration error) will compose fine at config time and produce
meaningless numbers at run time.

### A custom backbone

The top-level `module` group member owns the whole `LModule` config, so a new
backbone is a new file at the group root:

```yaml
# myconfigs/module/my_net.yaml
# @package module
_target_: radiologist.core.LModule
net:
  _target_: my_package.nets.MyBackbone
  num_classes: 3
  pretrained: true
trainable_layers:
  "": null            # dot-path into `net`; null unfreezes the whole submodule
priors: null
```

```bash
radiologist core hydra.searchpath=[file:///abs/path/to/myconfigs] module=my_net
```

`net` may be any `nn.Module` returning logits of shape `(B, num_classes)`. Two
things to keep in sync when you swap it:

- `module/metric/fbeta_score.yaml` interpolates `${module.net.num_classes}`, so
  your `net` config must expose a `num_classes` key under that exact path.
- `callbacks/attribution.yaml` and `callbacks/onnx_export.yaml` reference a
  `cam_target_layer` / `target_layer` by dot-path (`layer4` for ResNet). Override
  it for your architecture, or drop those callbacks.

### Other groups you can extend the same way

`module/optimizer`, `module/scheduler` (both `_partial_: true`), `callbacks`,
`loggers`, `strategy`, `debug`, and `datamodule` all follow the identical
pattern — add a file to your search-path directory with the matching
`# @package` header and select it by name.

### What is *not* config-swappable

`LModule` itself resolves `net`, `loss`, `metric`, `optimizer` and `scheduler`
from fixed key names, and the training/validation/test step bodies are not
extension points. Changing what happens inside a step means subclassing
`LModule` and pointing `module._target_` at your subclass.

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
