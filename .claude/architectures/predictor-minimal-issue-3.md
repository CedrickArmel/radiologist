## ✨ Explainer — Score-CAM saliency on top of classification

### Context

This slice replaces the skeleton stub for `Explainer.explain`. `Explainer` extends `Classifier`, so it inherits real loading and `predict` from #2; this issue adds only the saliency capability, driving the real `score_cam_with_session` in `cam.py` (reused verbatim) to GREEN-real. See `predictor-minimal-spec.md`.

> Requires: #1, #2.
> Blocks: #5 (app/CLI wire Explainer).

### User story

As a **clinical reviewer**, I want to **get a Score-CAM saliency map alongside the predicted class** so that **I can see which lung regions drove the prediction without standing up a separate classifier**.

### Acceptance criteria

- [ ] Given a deterministic ONNX path, `Explainer.from_path(model_path)` returns an `Explainer` that can also `predict` (inherited) — the same single model backs both.
- [ ] Given an image, `Explainer.explain(image)` returns an `Explanation` whose `saliency_map` is a 2-D array sized to the original image and whose values lie in `[0, 1]`.
- [ ] `explain`'s `predicted_class` matches the class `predict` returns for the same image and model.
- [ ] Given an image as a file path, a NumPy HWC uint8 array, or a PIL Image, `explain` returns a saliency map sized to that image's original dimensions.
- [ ] mypy clean; pytest green.

### Technical notes

- `predictor.py` — reuse the old `Predictor.explain` body verbatim; only the owning class changes to `Explainer` and the state field is `session` (not `det_session`).
- Saliency comes from `cam.score_cam_with_session(session, preprocessed, feature_maps, original_h, original_w)` — unchanged.

### Design notes

`Explainer(Classifier)` rather than `Explainer(BasePredictor)`: explanation is strictly additive over classification on the same model, so inheritance gives `predict` for free and guarantees the two outputs agree. This is the minimal-impact choice — no shared-collaborator extraction, no duplicated preprocessing.
