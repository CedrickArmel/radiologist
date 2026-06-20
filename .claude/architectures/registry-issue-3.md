## ✨ Core ONNX export → ExportResult, and WandbRegistry.promote

### Context

This slice executes the **Option B** split. It (a) introduces `export_onnx(...) -> ExportResult` in `radiologist-core` containing the pure ONNX work currently fused inside `promote_to_registry` (the `_CamWrapper`, dual deterministic/MC-dropout export, metadata stamping) with **no** W&B dependency; and (b) replaces the `WandbRegistry.promote` stub with the upload+link tail (promote.py:233-252). The caller resolves a checkpoint (#2), exports in core, then promotes the `ExportResult`. `ExportResult` itself is already defined in #1. See `registry-spec.md`. Requires: #1, #2. Target GREEN-real for both `export_onnx` and `WandbRegistry.promote`.

### User story

As an **ML engineer**, I want ONNX export decoupled from W&B upload so that I can export models without wandb installed and promote a finished `ExportResult` through the registry.

### Acceptance criteria

- [ ] Given a checkpoint path, `export_onnx` writes a deterministic ONNX and an MC-dropout ONNX into `local_dir` and returns an `ExportResult` carrying both paths, the `run_id`, `input_shape`, and `classes`.
- [ ] The deterministic model exposes `logits` and `feature_maps` outputs; the MC-dropout model preserves dropout in training mode and exposes `logits`.
- [ ] Both ONNX files carry embedded metadata (`classes`, `input_shape`, `precision`, `run_id`, framework; cam target layer on the deterministic model; `mc_dropout=true` on the MCD model).
- [ ] When `cam_target_layer` cannot be resolved against the loaded module, `export_onnx` raises `AttributeError` naming the missing segment.
- [ ] `export_onnx` runs with wandb absent (raises `RuntimeError` only if `onnx` is missing).
- [ ] Given an `ExportResult`, `WandbRegistry.promote` uploads both ONNX files, links each to the given collection under the given alias, and returns the qualified name of the linked MC-dropout artifact.
- [ ] When wandb is not installed, `promote` raises `RuntimeError`.
- [ ] mypy clean; pytest green.

### Interface contracts

##### `radiologist-core/src/radiologist/core/registry/export.py`

```python
def export_onnx(
    ckpt_path: str,
    input_shape: Tuple[int, ...],
    classes: List[str],
    cam_target_layer: str,
    local_dir: str,
    run_id: str,
    precision: str,
    opset: int = 18,
) -> "ExportResult":
    # contract: loads LModule from ckpt_path, exports deterministic + MCD ONNX
    # into local_dir with embedded metadata, returns ExportResult. raises
    # AttributeError if cam_target_layer missing; RuntimeError if onnx missing.
```

### Module layout

```
radiologist-core/src/radiologist/core/registry/
└── export.py   # export_onnx + _CamWrapper + _resolve_layer + _set_metadata_props (moved from promote.py)
```

### Out of scope

- Removing the old `promote_to_registry`/`pull_checkpoint` and rewiring `__all__` (#5).
- Alias management (#4).

### Technical notes

- `export.py` imports `ExportResult` from `radiologist.registry.base`; core's `registry` extra must add a dependency on `radiologist-registry`. Keep core free of any direct `import wandb` in `export.py`.
- Move `_CamWrapper`, `_resolve_layer`, `_set_metadata_props` out of `promote.py` into `export.py`; use `model.train(mode=False)` (never `.eval()`).
- `WandbRegistry.promote` reuses the artifact naming `model-{run_id}` / `model-{run_id}-mcd` from promote.py:234-249; mock only `wandb.init`/`wandb.Artifact`/`Api` in tests.

### Design notes

`ExportResult.run_id` and `precision` are passed into `export_onnx` rather than re-read from W&B inside core — this is what keeps the export path W&B-free. The caller (who already resolved the artifact in #2) reads `logged_by().id` and `config["trainer"]["precision"]` from the resolved run and hands them in. This is the minimal seam that lets core stay a pure exporter.
