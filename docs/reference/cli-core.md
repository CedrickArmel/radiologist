# Core (Training) CLI Reference

`radiologist core` is the training command group of the unified
`radiologist` CLI, wrapping `radiologist.core.train:train` as a Hydra entry
point
(`@hydra.main(config_path="pkg://radiologist.core.configs", config_name="train")`).
Unlike the other three groups, `core` has **no subcommand** — it *is* the
command, invoked as `radiologist core <hydra overrides>`. This is easy to
miss given `etl`, `registry`, and `infer` all expose subcommands. `core`
ships in `radiologist-cli`'s hard dependencies, so no extra is needed:

```bash
pip install radiologist-cli
```

Because it's a Hydra entry point, every parameter is a config key overridden
with `key=value` syntax (or added with `+key=value` for a key not already in
the tree), rather than `--flag value` options, and `--multirun` sweeps and
`--config-name`/`--config-path` overrides work out of the box.
`radiologist core --help` prints the full composed config tree (Hydra's own
generated help) for `train.yaml`. See [`config-core.md`](config-core.md)
for the complete key reference across every group (`trainer`, `module`,
`datamodule`, `callbacks`, `loggers`, `strategy`, `debug`, `hydra`) — it is
not duplicated here.

```bash
# Train with defaults
radiologist core

# Override trainer/module/datamodule keys
radiologist core trainer.max_epochs=30 module/optimizer=adamw datamodule.batch_size=16

# Evaluate a checkpoint instead of training
radiologist core --config-name eval ckpt_path=/path/to/best.ckpt
```

`--config-name eval` is the only way to reach `eval.yaml` (`train: false`,
`test: true`, `ckpt_path: ???` mandatory) instead of the default
`train.yaml` — `core` folds its "alternate mode" into a Hydra config-name
switch rather than a separate subcommand.

## Emitted record

On success, `radiologist core` emits a single keyed record via
`radiologist.utils.cli.emit` — `kv` (default, `key=value` lines), `json`, or
`yaml`, selected with the global `--output`/`-o` flag or the
`RADIOLOGIST_OUTPUT` environment variable, e.g.
`radiologist --output json core trainer.max_epochs=5`.

Emitted keys: `run_id`, `best_ckpt_path`, `det_onnx_path`, `mcd_onnx_path`,
`det_qualified_name`, `mcd_qualified_name`.

| Key | Description |
|---|---|
| `run_id` | The active W&B run ID, or `null` when no W&B run is active (e.g. `loggers: null`, as `eval.yaml` sets). |
| `best_ckpt_path` | Path to the best checkpoint tracked by the `ModelCheckpoint` callback, or `null` if unavailable. |
| `det_onnx_path` | Path to the exported deterministic ONNX model, or `null` unless `OnnxExportCallback` (opt-in, not part of `callbacks: default`) exported one and a W&B run is active. |
| `mcd_onnx_path` | Path to the exported MC-Dropout ONNX model — `null` under the same conditions as `det_onnx_path`. |
| `det_qualified_name` | `{entity}/{project}/model-{run_id}:best` once `det_onnx_path` is resolved, else `null`. |
| `mcd_qualified_name` | `{entity}/{project}/model-{run_id}-mcd:best` once `mcd_onnx_path` is resolved, else `null`. |

The det/mcd ONNX paths and their registry-qualified names are not returned
by `train()` itself — they are a side effect of
`OnnxExportCallback.on_fit_end` (silent no-op without an active W&B run or a
best checkpoint), reconstructed here from the same naming convention the
callback and the registry uploader use, gated on the exported file actually
existing on disk.

## Exit codes

Exceptions are mapped to a process exit code via
`radiologist.utils.cli.exit_code_for`: `2` for `FileNotFoundError` (e.g. a
missing `ckpt_path`), `1` for any other failure. A failing run prints
`Error: {message}` on stderr and emits no result record.

`radiologist core` additionally forces `HYDRA_FULL_ERROR=1` in the process
environment for the duration of the run (restored to its previous value
afterward). Without it, Hydra's own `run_and_report()` swallows every
exception raised inside the training entry point and always exits `1`,
which would flatten the exit-code-per-exception-type contract above to a
single code regardless of cause — a non-obvious behavior worth knowing if
you're scripting around the exit code.

## Python API equivalent

`radiologist core` is a thin CLI wrapper over `radiologist.core.train`, so a
training (or evaluation) run can be driven directly without Hydra or the
CLI, by composing the config yourself:

```python
from hydra import compose, initialize_config_module

from radiologist.core.train import train

with initialize_config_module(config_module="radiologist.core.configs", version_base="1.3"):
    cfg = compose(config_name="train", overrides=["trainer.max_epochs=5"])

metric_dict, object_dict = train(cfg)
```

Pass `config_name="eval"` (plus `overrides=["ckpt_path=..."]`) to run the
evaluation-only variant instead. See the [API Reference](api-core.md) and
[`config-core.md`](config-core.md) for the full config schema `train()`
expects.
