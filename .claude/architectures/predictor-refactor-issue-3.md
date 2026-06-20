## ✨ `Classifier.predict`

### Context

Replaces the `Classifier.predict` skeleton stub with the real deterministic-inference logic, ported from the legacy `Predictor.predict`. The prior-correction helper that `predict` uses is internal to this slice — it is built top-down through `predict` and is not a separate issue. After this slice a caller can build a `Classifier` from a single deterministic ONNX model and classify images. See the epic spec for the hierarchy. Requires: #1, #2. Target GREEN-real: no `NotImplementedError` reachable through `Classifier.from_path(...).predict(...)`.

### User story

As a **clinical integrator**, I want to load only the deterministic classifier and get per-class probabilities so that the lightest possible model serves a prediction.

### Acceptance criteria

- [ ] Given a deterministic ONNX model and an image, `predict` returns probabilities keyed by every class name, summing to ~1.0, plus the argmax class as `predicted_class`.
- [ ] Given a `deployment_prior`, the returned probabilities are the prior-corrected, renormalized distribution (and override any embedded training prior).
- [ ] Given a model with an embedded `training_prior` and no `deployment_prior`, the embedded prior is applied.
- [ ] Given a model with no embedded prior and no `deployment_prior`, raw softmax probabilities are returned.
- [ ] A `Classifier` built from only the deterministic path exposes no uncertainty capability (it is not an `MCDropoutPredictor`) — classification needs no MC-Dropout model.
- [ ] mypy clean; pytest green.

### Technical notes

- Port the softmax + prior logic from legacy `Predictor.predict` (`predictor.py:208-250`) and `_apply_prior_correction` (`predictor.py:113-133`). The prior helper stays a private function in `classifier.py`, reached only through `predict`.
- Read `classes` / `input_shape` from `self.metadata`; preprocess via `preprocess_image` (#2); run the session for `["logits"]`.
- The existing `test_predict.py` exercises this behavior through the old `Predictor`; this slice must satisfy the same observable contract through `Classifier.predict`. Keep the legacy `Predictor` working too (removed in #9), so both paths stay green.

### Design notes

`predict` lives on `Classifier`, not `BasePredictor`, so that a future non-classifying capability could subclass `BasePredictor` without inheriting a classification method it does not want. `Explainer` subclasses `Classifier` precisely to reuse `predict` for the predicted-class label in `explain` (#4).
