# Training (Hydra) Configuration Reference

`radiologist-core` composes its training/evaluation run entirely from a
[Hydra](https://hydra.cc) config tree at `radiologist-core/src/radiologist/core/configs/`.
The Hydra entry point is `radiologist.cli.groups.core:train_main`, shipped in the
`radiologist-cli` package and reached as the `core` group of the unified
`radiologist` command. It loads `configs/train.yaml` by default, or another
top-level config via `--config-name` (e.g. `eval`).

Everything after `core` is forwarded to Hydra verbatim, so every group below can
be overridden on the command line:

```bash
radiologist core \
    trainer.max_epochs=30 \
    module/optimizer=adamw \
    datamodule.batch_size=16
```

```bash
radiologist core --config-name eval ckpt_path=/path/to/checkpoint.ckpt
```

The `core` group backs onto a hard dependency, so plain `pip install radiologist-cli`
is enough — unlike `etl`, `registry`, and `infer`, it needs no extra.

## Top-level singletons

These are `# @package _global_` configs merged directly into the root
namespace — not selectable groups, just always-present pieces of the tree.

| File | Purpose |
|---|---|
| `train.yaml` | Root config for a training run. Wires every group below via its `defaults` list (`paths`, `extras`, `hydra: default`, `trainer`, `module: resnet50` + its `loss`/`scheduler`/`metric`/`optimizer` sub-groups, `datamodule: default`, `strategy: auto`, `loggers: wandb`, `callbacks: default`). Sets `stage: train`, `train: true`, `test: true`, `seed: 42`, and `optimized_metric: best_val_score` — the key `main()` reads to return the HPO objective value. |
| `eval.yaml` | Evaluation-only variant of `train.yaml`. Same group wiring except `loggers: null` and no `module/scheduler`/`module/optimizer` (not needed without a fit loop). Sets `stage: eval`, `train: false`, `test: true`, `ckpt_path: ???` (mandatory — must be supplied via `+ckpt_path=...` or CLI override), and `optimized_metric: test_score`. |
| `trainer.yaml` | Instantiates `lightning.pytorch.trainer.Trainer`. Key defaults: `accelerator: auto`, `devices: auto`, `precision: 32-true`, `max_epochs: -1` (unbounded — rely on early stopping / external scheduling), `min_epochs: 1`, `gradient_clip_val: 2` with `gradient_clip_algorithm: norm`, `deterministic: true`, `use_distributed_sampler: false` (the datamodule handles node/worker splitting itself via WebDataset), `default_root_dir: ${paths.output_dir}`. |
| `paths.yaml` | Derives `root_dir` from `hydra:runtime.cwd`, and `data_dir`/`log_dir`/`work_dir` relative to it; `output_dir` is Hydra's own per-run `runtime.output_dir` (the timestamped folder under `outputs/` or `multirun/`). Every other config that writes to disk (checkpoints, W&B `save_dir`, prediction dumps) points at `${paths.output_dir}`. |
| `extras.yaml` | Cross-cutting run behavior consumed by `radiologist.utils.ml.extras` before `train()` runs: `ignore_warnings: true`, `enforce_tags: true` (fails fast if `tags` is unset — helps keep W&B runs searchable), `print_config: true` (prints the resolved config tree at run start). |

## `callbacks/`

Selected as a group (default: `callbacks: default`), which itself is a
`defaults` list composing six named callback configs together (all under
`# @package callbacks`, keyed by name so they merge into one dict Hydra
instantiates as a callback list).

| File | Purpose |
|---|---|
| `default.yaml` | Composes `best_metric`, `wandb_summary`, `attribution`, `model_checkpoint`, `lr_monitor`, `onnx_export` into the callback list used by `train.yaml`/`eval.yaml`. |
| `best_metric.yaml` | `radiologist.core.BestMetricCallback` — `monitor: val_score`, `mode: max`. Tracks the best validation score across epochs (see [`BestMetricCallback`][radiologist.core.BestMetricCallback]). |
| `wandb_summary.yaml` | `radiologist.core.WandbDefineSummaryCallback` — `monitor: val_score`, `mode: max`. Configures the W&B run-summary panel (see [`WandbDefineSummaryCallback`][radiologist.core.WandbDefineSummaryCallback]). |
| `attribution.yaml` | `radiologist.core.AttributionCallback` — `target_layer: layer4`, runs every validation epoch (`every_n_val_epochs: 1`) on 1-in-10 batches (`every_n_batches: 10`), `save_to_file: false` (W&B-only logging), Integrated Gradients disabled (`ig_n_steps: null`). See [`AttributionCallback`][radiologist.core.AttributionCallback]. |
| `model_checkpoint.yaml` | `lightning.pytorch.callbacks.ModelCheckpoint` — writes to `${paths.output_dir}/checkpoints`, filename pattern `epoch={epoch}-step={step}`, monitors `val_score` in `max` mode, keeps only the best (`save_top_k: 1`) plus `save_last: true`. |
| `lr_monitor.yaml` | `lightning.pytorch.callbacks.LearningRateMonitor` — logs the LR at `step` granularity. |
| `onnx_export.yaml` | `radiologist.core.OnnxExportCallback` — part of `default.yaml`'s composition. It is a silent no-op without an active W&B run and a best checkpoint, which is why it is safe to leave in the default list. `input_shape: [1, 3, 224, 224]`, `classes: ["normal", "viral", "opacity"]` (placeholder values — override to match the actual class list), `cam_target_layer: layer4`, `opset: 18`. Exports ONNX artifacts at fit end (see [`OnnxExportCallback`][radiologist.core.OnnxExportCallback]). |

## `module/`

Selected as a group (default: `module: resnet50`); its sub-groups
(`optimizer`, `scheduler`, `loss`, `metric`) are selected independently and
merged under the `module` package.

| File | Purpose |
|---|---|
| `resnet50.yaml` | Root module config, `_target_: radiologist.core.LModule`. Backbone `net` is `torchvision.models.resnet50` with `num_classes: 3` (hardcoded here; note `module/metric/fbeta_score.yaml` instead interpolates `${module.net.num_classes}`, so keep both in sync if this changes). `trainable_layers: null` and `priors: null` — full weight re-initialisation from scratch, no fine-tuning subset, no calibrated bias init (see [`LModule`][radiologist.core.LModule] `setup()` behavior). |
| `optimizer/adamw.yaml` | `torch.optim.AdamW` partial factory — `lr: 1e-3`, `weight_decay: 1e-2`, default betas/eps. |
| `scheduler/sequential.yaml` | `radiologist.utils.ml.sequential_scheduler` partial factory chaining `LinearLR` (warmup, `start_factor: 0.1` over `total_iters: 500` steps) into `CosineAnnealingLR` (`T_max: 10000`), switching at `milestones: [500]`. |
| `loss/focal_loss.yaml` | `radiologist.core.FocalLoss` — `gamma: 2`, `alpha: 1`, `reduction: mean`, `use_softmax: true`, `to_onehot_y: true` (targets arrive as class indices and are one-hot encoded internally). |
| `metric/fbeta_score.yaml` | `torchmetrics.classification.MulticlassFBetaScore` partial factory — `beta: 1.0` (F1), `average: macro`, `num_classes` interpolated from `${module.net.num_classes}`, `compute_on_cpu: true`, `sync_on_compute: true`. |

## `datamodule/`

| File | Purpose |
|---|---|
| `default.yaml` | `radiologist.core.WebDatasetDataModule` (see [`WebDatasetDataModule`][radiologist.core.WebDatasetDataModule]). Points at GCS shard/manifest URIs (`shard_root`, `split_manifest_uri`), `batch_size: 10`, `seed: ${seed}`. `label_map` collapses four raw ETL labels (`normal`, `viral_pneumonia`, `covid`, `lung_opacity`) into three model classes (`healthy`, `viral`, `opacity`); `classes` fixes their order. `class_weights`/`priors` are `null` — both are auto-computed from shard counts at `setup()`. `train_transform` applies grayscale→RGB, resize, random-resized-crop, color jitter, then normalizes with `mean: [128]`/`std: [65]`; `eval_transform` skips the augmentation steps. `train_loader`/`eval_loader` are partial `webdataset.WebLoader` factories with seeded `worker_init_fn`/`generator` for reproducibility. |

## `debug/`

Not selected by default (`debug: null` in `train.yaml`/`eval.yaml`); opt in
with e.g. `+debug=fast_dev_run`. All are `# @package _global_` overlays.

| File | Purpose |
|---|---|
| `base.yaml` | Shared base the other four `debug/` configs extend via `defaults: [base, _self_]`. Sets `stage: debug`, disables `extras.ignore_warnings`/`extras.enforce_tags`, sets Hydra's root logger to `DEBUG`, forces `trainer.max_epochs: 1` and `trainer.detect_anomaly: true`, and clears `optimized_metric`. |
| `fast_dev_run.yaml` | Sets `trainer.fast_dev_run: true` (one train/val/test batch each) and disables `callbacks`/`loggers` entirely — the quickest smoke test of the full loop. |
| `barebones.yaml` | Sets `trainer.barebones: true` (Lightning's minimal-overhead mode — disables checkpointing, progress bar, model summary, sanity-check steps, anomaly detection, and the profiler) and also disables `callbacks`/`loggers`. Use to isolate raw step throughput from logging/checkpointing overhead. |
| `limit.yaml` | Restricts the run to small dataset fractions (`limit_train_batches: 0.01`, `limit_val_batches`/`limit_test_batches: 0.05`) and disables `model_checkpoint`/`early_stopping`/`exception_checkpoint` callback entries. **Flagged as unclear**: `limit.yaml` and `overfit.yaml` both null out `callbacks.early_stopping` and `callbacks.exception_checkpoint`, but no `early_stopping.yaml` or `exception_checkpoint.yaml` file exists under `callbacks/` and neither key appears in `callbacks/default.yaml` — these look like overrides anticipating callbacks that either haven't been added yet or were renamed/removed; the override is inert until such keys exist. |
| `overfit.yaml` | Deliberate-overfit smoke test — `trainer.max_epochs: 100`, `trainer.overfit_batches: 4` (repeatedly trains on the same 4 batches to sanity-check the model can memorize), `detect_anomaly: false`. Same `early_stopping`/`exception_checkpoint` callback-nulling caveat as `limit.yaml` above. |

## `loggers/`

| File | Purpose |
|---|---|
| `wandb.yaml` | Selected by default in `train.yaml` (`loggers: wandb`; `eval.yaml` uses `loggers: null`). `lightning.pytorch.loggers.WandbLogger` — `project: radiologist`, `entity: theradiologist-liora-liora`, `save_dir: ${paths.output_dir}`, `offline: false`, `log_model: false` (checkpoints are exported to ONNX and pushed to the registry separately, not logged as raw W&B artifacts here), `tags: ${tags}` (interpolated from the root config; `extras.enforce_tags` will fail the run if this is left unset). |

## `strategy/`

| File | Purpose |
|---|---|
| `auto.yaml` | Selected by default (`strategy: auto`). A single scalar value (not a nested config) forwarded to `trainer.strategy` / Lightning's `Trainer(strategy=...)` — `"auto"` lets Lightning pick single-device vs. DDP based on the detected accelerator/device count. |

## `hydra/`

| File | Purpose |
|---|---|
| `default.yaml` | Selected by default (`hydra: default`). Configures Hydra's own run/sweep output layout: single runs go to `outputs/<date>/<time>/`, multirun (sweep) jobs go to `multirun/<date>/<time>/<job-num>/`. Sets `hydra.job.chdir: true`, so `train()`/`main()` execute with the process CWD already inside that per-run output directory — this is what `paths.root_dir: ${hydra:runtime.cwd}` and `paths.output_dir: ${hydra:runtime.output_dir}` resolve against. |

## Bring your own Hydra config

The config tree above ships **inside the installed wheel** — `train_main` is
declared with `config_path="pkg://radiologist.core.configs"`, not a filesystem
path. You extend it by giving Hydra an additional search path rather than by
editing the installed package.

```bash
# as a config override
radiologist core hydra.searchpath=[file:///abs/path/to/myconfigs] module/loss=my_loss

# as the equivalent Hydra CLI flag
radiologist core --config-dir /abs/path/to/myconfigs module/loss=my_loss
```

Mirror the packaged group layout inside your directory:

```
myconfigs/
├── module/
│   ├── my_net.yaml            # -> module=my_net
│   ├── loss/my_loss.yaml      # -> module/loss=my_loss
│   ├── metric/my_metric.yaml  # -> module/metric=my_metric
│   ├── optimizer/my_opt.yaml  # -> module/optimizer=my_opt
│   └── scheduler/my_sched.yaml
├── callbacks/my_callbacks.yaml
└── datamodule/my_datamodule.yaml
```

Each file needs the same `# @package` header its packaged siblings use:

| Group | Required header |
|---|---|
| `module`, `module/loss`, `module/metric`, `module/optimizer`, `module/scheduler` | `# @package module` |
| `callbacks/*` | `# @package callbacks` |
| `datamodule/*`, `trainer.yaml`, `debug/*` | `# @package _global_` |

Omitting the header is the most common failure: Hydra infers the package from
the directory name, your `loss:` key lands at `module.loss.loss`, and `LModule`
— which reads `cfg.loss` — never sees it. The symptom is an instantiation error
deep in the run, not a config error at compose time. Check with
`radiologist core --cfg job` before launching anything expensive; it prints the
fully composed config and exits.

A minimal custom loss:

```yaml
# myconfigs/module/loss/my_loss.yaml
# @package module
loss:
  _target_: my_package.losses.LabelSmoothedCE
  smoothing: 0.05
```

`_target_` may point at any importable class — your package does not have to be
part of this workspace, it only has to be on `sys.path` of the interpreter
running `radiologist`.

See [radiologist-core/README.md](../pkg/core.md) for the per-key contract each
group member must satisfy (which entries need `_partial_: true`, what `LModule`
calls each collaborator with, and the `${module.net.num_classes}` interpolation
you must keep intact when swapping the backbone).
