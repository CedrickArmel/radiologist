## ✨ `Explainer.explain`

### Context

Replaces the `Explainer.explain` skeleton stub with the real Score-CAM logic, ported from legacy `Predictor.explain`. `Explainer` subclasses `Classifier`, so it inherits `predict` and reuses it for the predicted-class label. The full Score-CAM pass (`score_cam_with_session`) is reached only through `explain` and is not a separate issue; the stateless `score_cam` re-export stays public and is exercised through its own existing public test. Requires: #1, #2, #3. Target GREEN-real: no `NotImplementedError` reachable through `Explainer.from_path(...).explain(...)`.

### User story

As a **radiologist reviewer**, I want a saliency map plus the predicted class for an image so that I can see which lung regions drove the abnormal/normal decision.

### Acceptance criteria

- [ ] Given a deterministic ONNX model and an image, `explain` returns a saliency map (2-D float array) with values in `[0, 1]` sized to the original image dimensions, plus the predicted class label.
- [ ] The predicted class returned by `explain` matches the class `predict` returns for the same image (same underlying classification).
- [ ] An `Explainer` can also classify (it is a `Classifier`): `predict` works on the same instance.
- [ ] mypy clean; pytest green.

### Technical notes

- Port from legacy `Predictor.explain` (`predictor.py:252-294`): read original `H`/`W` from the source image, preprocess via `preprocess_image` (#2), run the session for `["logits", "feature_maps"]`, call `score_cam_with_session` from `cam.py`, and resolve the predicted label.
- Reuse the inherited `Classifier.predict` to derive `predicted_class` rather than re-deriving the argmax inline — this is the reason `Explainer` extends `Classifier`.
- `test_score_cam.py` currently drives this through old `Predictor.explain`; the same observable contract must hold through `Explainer.explain`.
- `cam.py` is unchanged — only its call site moves into `explainer.py`.
