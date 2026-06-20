## ✨ `from_registry` delegation to `WandbRegistry`

### Context

Replaces the `BasePredictor.from_registry` skeleton stub with real delegation to the `radiologist-registry` package: it pulls the model via `WandbRegistry().pull(artifact_path, local_dir)` then calls `cls.from_path(local_path)`. Because `from_registry` lives on `BasePredictor`, every capability subclass inherits registry-backed construction. The `pull_model` bridge is **not** reintroduced — delegation is direct. Requires: #1, #2 (needs a working `from_path`). Sequences after registry epic #85–90. Target GREEN-real: no `NotImplementedError` reachable through `<AnyCapability>.from_registry(...)`.

### User story

As an **ops engineer**, I want to construct any inference capability straight from a W&B artifact path so that deployments pull the exact registered model without a separate download step.

### Acceptance criteria

- [ ] Given a resolvable artifact path and a local directory, `Classifier.from_registry` returns a `Classifier` whose `predict` works on an image (the artifact's ONNX file was pulled and loaded).
- [ ] `Explainer.from_registry` and `MCDropoutPredictor.from_registry` likewise return the correct concrete subclass, loaded from the pulled model.
- [ ] When the `registry` extra (wandb) is not installed, `from_registry` raises `RuntimeError` naming the `registry` extra.
- [ ] mypy clean; pytest green.

### Technical notes

- Import `WandbRegistry` from `radiologist.registry` inside `from_registry` (lazy import) so the inference package imports cleanly without the `registry` extra; guard on the same `_wandb` sentinel pattern from `optional.py` to raise the `RuntimeError` naming `registry`.
- `from_registry` must return `cls(...)`-typed instances — calling it on `Explainer` returns an `Explainer`, not a `BasePredictor`. Use `cls.from_path(...)` so the classmethod's `cls` flows through.
- `WandbRegistry().pull(artifact_path, local_dir) -> str` returns the local model path (registry epic #85–90, issue #2). In tests, mock only the registry boundary (the `WandbRegistry` / wandb API) — a true process boundary — and feed `from_path` the real fixture ONNX produced by `conftest.py`.
- Coordinate with registry epic issue #5, which previously rewired the old `Predictor.from_registry`; this issue is the final authority for the new hierarchy. The cleanup (#9) ensures no `pull_model` reference survives.

### Design notes

`from_registry` delegates rather than wrapping: no `pull_model` shim remains in the inference package. Model resolution/download is entirely the registry package's responsibility; the inference package's only job is to turn a local ONNX path into a typed capability. This keeps the inference package free of any W&B artifact-resolution logic.
