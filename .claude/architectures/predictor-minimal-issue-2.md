## ✨ Classifier — deterministic classification + shared loading

### Context

This slice replaces the skeleton stubs for `BasePredictor.from_path`, `BasePredictor.from_registry`, and `Classifier.predict`. It owns the shared loading seam every other subclass inherits — so it is built first among the slices. Implement top-down through the real `_read_metadata`, `_preprocess_image`, and `_apply_prior_correction` helpers (reused verbatim) and a real `ort.InferenceSession`, reaching GREEN-real: no `NotImplementedError` reachable through `Classifier`'s public API. See `predictor-minimal-spec.md` for architecture context.

> Requires: #1.
> Blocks: #3 (Explainer extends Classifier), #5 (app/CLI wire Classifier).

### User story

As a **deployment engineer**, I want to **load a single deterministic ONNX model and classify an image** so that **a classification-only service loads no explanation or uncertainty machinery**.

### Acceptance criteria

- [ ] Given a deterministic ONNX path, `Classifier.from_path(model_path)` returns a `Classifier` whose `predict(image)` yields a `Prediction` with one probability per model class and a `predicted_class` that is one of those classes.
- [ ] Given an image as a file path, a NumPy HWC uint8 array, or a PIL Image, `predict` returns equivalent class probabilities (input form does not change the contract).
- [ ] Given a `deployment_prior`, `predict` returns probabilities re-weighted by that prior and renormalized to sum to 1.
- [ ] Given a model with an embedded `training_prior` and no `deployment_prior`, `predict` applies the embedded prior; given neither, it returns raw softmax probabilities.
- [ ] When `from_path` is given a path that does not exist, it raises `FileNotFoundError`.
- [ ] When `from_registry` is called and the `registry` extra (wandb) is absent, it raises `RuntimeError` whose message names the `registry` extra.
- [ ] When `from_registry` is called with the extra present, it returns a `Classifier` loaded from the artifact resolved by the registry (registry interaction mocked at the `WandbRegistry.pull` boundary).
- [ ] mypy clean; pytest green.

### Technical notes

- `predictor.py` — reuse the existing softmax + `_apply_prior_correction` logic from the old `Predictor.predict` verbatim; only the class it lives on changes.
- `_PredictorState` now holds a single `session` (renamed from `det_session`); update the helper call sites accordingly.
- `from_registry` delegates to `WandbRegistry.pull(artifact_path, local_dir)` from the registry epic — do not reintroduce a `pull_model` bridge.

### Design notes

`from_path` and `from_registry` live on `BasePredictor` and return `cls(...)` so subclasses inherit loading for free — the minimal-impact choice over duplicating a loader per subclass. The deterministic-vs-stochastic distinction is encoded purely by which subclass the caller instantiates, not by a flag on the state.
