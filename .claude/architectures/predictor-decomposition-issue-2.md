## ✨ Classifier — deterministic classification with model loading

### Context

First slice. Replaces the skeleton stubs for `BasePredictor` (model loading, metadata, `from_path`, `from_registry`) and `Classifier.predict`. Loading and preprocessing are the seams every other slice builds on, so they are implemented here and exercised through `Classifier`'s public API — no helper is tested in isolation. Reaches GREEN-real for classification: no `NotImplementedError` reachable through `Classifier.from_path(...).predict(...)`. Requires #1. See `predictor-decomposition-spec.md`.

### User story

As an **inference user**, I want to **load only a deterministic ONNX model and classify a chest X-ray** so that **I get per-class probabilities and a predicted label without dragging Score-CAM or MC-Dropout code**.

### Acceptance criteria

- [ ] Given a deterministic ONNX model and an image (file path, HWC uint8 array, or PIL image), `Classifier.from_path(model_path).predict(image)` returns per-class probabilities summing to ~1.0 and a `predicted_class` equal to the argmax class.
- [ ] When a `deployment_prior` is supplied, the returned probabilities differ from the raw softmax (prior correction is applied) and still sum to ~1.0.
- [ ] When the model embeds a `training_prior` and no `deployment_prior` is given, the embedded prior is applied; when neither is present, raw softmax probabilities are returned.
- [ ] When `from_path` is given a path that does not exist, it raises `FileNotFoundError`.
- [ ] `Classifier.from_path(...)` returns an instance whose type is `Classifier` (the `cls` it was called on), and the same `predict` result keys match the model's class set.
- [ ] When the registry dependency is unavailable, `Classifier.from_registry(artifact_path, local_dir)` raises `RuntimeError` whose message names the `registry` extra.
- [ ] When the registry dependency is available, `Classifier.from_registry(...)` returns a `Classifier` that produces a `Prediction` with the same probability keys as a `Classifier` loaded from the same local model via `from_path`.
- [ ] mypy clean; pytest green.

### Technical notes

- Migrate the deterministic-inference body and the `_preprocess_image` / `_apply_prior_correction` / `_read_metadata` seams from the old `Predictor` into `BasePredictor` / `Classifier`. `from_path`/`from_registry` are classmethods returning `cls` so subclasses inherit them.
- `test_predict.py` and `test_registry.py` are re-pointed from `Predictor` to `Classifier` in this slice (they own these behaviors). Mock only the registry/wandb boundary in `from_registry` tests, never the ONNX session for the local-load tests.
- Use `model.train(mode=False)` semantics only in PyTorch code — there is none here (ONNX runtime), so no concern.
