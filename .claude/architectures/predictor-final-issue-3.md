## ✨ Implement `Explainer`

### Context

This slice replaces the skeleton stub for `Explainer.explain`. `Explainer` subclasses
`Classifier`, so it inherits a fully working `predict` and the shared `BasePredictor`
constructors and helpers — this issue adds only the Score-CAM explanation path. Implement
top-down to GREEN-real: no `NotImplementedError` reachable through
`Explainer.from_path(...).explain(...)`. See the epic spec for architecture.

**Blocked by:** #1 (skeleton) and #2 (inherits the implemented `Classifier`).

### User story

As a **clinician reviewing a prediction**, I want to **get a saliency map highlighting the
regions that drove the classification** so that **I can visually assess where the model
focused on the X-ray**.

### What to implement

**`explainer.py`** — port the monolith's `explain` onto `Explainer`:

- load the original image (str / ndarray / PIL → RGB) and capture its `(original_w,
  original_h)`;
- read `classes` and `input_shape` from metadata; preprocess via the inherited
  `_preprocess_image`;
- run the det session for `["logits", "feature_maps"]`;
- compute saliency with `score_cam_with_session(session, preprocessed, feature_maps,
  original_h, original_w)` from `cam.py`;
- compute the predicted class from the logits (softmax argmax) and return
  `Explanation(saliency_map=saliency, predicted_class=classes[argmax])`.

Reuse the inherited preprocessing/metadata access rather than re-reading state directly
where the base already exposes it. Do not duplicate `predict` — it is inherited.

### Tests

Own the explanation behavioral tests (migrate the relevant cases from `test_score_cam.py`
that exercise the class path; the stateless `score_cam` helper tests stay as-is). Drive
through `Explainer.explain` with the real `det_onnx_path` fixture.

- Given a det model and an image, `Explainer.from_path(...).explain(image)` returns an
  `Explanation` whose `saliency_map` is a numpy array sized to the **original** image
  resolution (height × width of the input image, not the model input shape).
- The `predicted_class` returned by `explain` is one of the model's classes and agrees
  with the `predicted_class` returned by `predict` on the same image.
- `saliency_map` values lie within the normalized range produced by Score-CAM (all values
  in `[0, 1]`).
- An `Explainer` instance also answers `predict` (inherited) returning a `Prediction` —
  asserted through the public API, confirming the inheritance contract holds.

### Acceptance criteria

- [ ] Given a det model and an image, `explain` returns an `Explanation` whose `saliency_map` matches the original image resolution.
- [ ] The `predicted_class` from `explain` agrees with the `predicted_class` from `predict` on the same image.
- [ ] `saliency_map` values are all within `[0, 1]`.
- [ ] An `Explainer` also serves `predict` (inherited) and returns a `Prediction`.
- [ ] mypy clean; pytest green.
