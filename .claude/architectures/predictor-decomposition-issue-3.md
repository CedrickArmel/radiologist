## ✨ Explainer — Score-CAM saliency on a single deterministic model

### Context

Second slice. Replaces the skeleton stub for `Explainer.explain`. `Explainer` subclasses `Classifier`, so it reuses the loading, preprocessing, and `predict` already made real in #2 and adds only the saliency capability driven through the deterministic session. Reaches GREEN-real for explanation: no `NotImplementedError` reachable through `Explainer.from_path(...).explain(...)`. Requires #2 (subclasses `Classifier`). See `predictor-decomposition-spec.md`.

### User story

As an **inference user**, I want to **load a single deterministic model and get a Score-CAM saliency map** so that **I can visualize what drove the classification without loading a separate explanation model**.

### Acceptance criteria

- [ ] Given a deterministic ONNX model and an image, `Explainer.from_path(model_path).explain(image)` returns an `Explanation` whose `saliency_map` is a 2-D array with all values in `[0, 1]`.
- [ ] The returned `saliency_map` spatial dimensions match the original image height and width (the map is upsampled to input resolution).
- [ ] The `predicted_class` on the returned `Explanation` equals the argmax class for that image (consistent with what `predict` would return).
- [ ] An `Explainer` instance also answers `predict(...)` identically to a `Classifier` loaded from the same model (inheritance preserved).
- [ ] mypy clean; pytest green.

### Technical notes

- Migrate the `explain` body from the old `Predictor`, including the `score_cam_with_session` call against the deterministic session; do not duplicate preprocessing — reuse the inherited seam.
- `test_score_cam.py` is re-pointed from `Predictor.explain` to `Explainer.explain` in this slice.
