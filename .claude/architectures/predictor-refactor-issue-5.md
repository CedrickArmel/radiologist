## ✨ `MCDropoutPredictor.predict_with_uncertainty`

### Context

Replaces the `MCDropoutPredictor.predict_with_uncertainty` and `mc_dropout_predict` skeleton stubs with the real MC-Dropout logic, ported from the legacy `Predictor.predict_with_uncertainty` and the standalone `mc_dropout_predict`. `MCDropoutPredictor` subclasses `BasePredictor` directly and loads only the MC-Dropout model — it carries no deterministic/explanation session. Requires: #1, #2. Target GREEN-real: no `NotImplementedError` reachable through `MCDropoutPredictor.from_path(...).predict_with_uncertainty(...)` or `mc_dropout_predict(...)`.

### User story

As a **safety reviewer**, I want mean probabilities, per-class spread, and predictive entropy from a stochastic model so that I can flag low-confidence predictions for human review.

### Acceptance criteria

- [ ] Given an MC-Dropout ONNX model and an image, `predict_with_uncertainty` returns mean probabilities and per-class std keyed by every class name, plus a non-negative predictive entropy and the pass count.
- [ ] Running with a larger `n_passes` increases the number of stochastic passes aggregated (the returned `n_passes` reflects the argument).
- [ ] The standalone `mc_dropout_predict`, given a session and a preprocessed array, returns the same aggregated `UncertaintyResult` shape — it stays a public stateless helper.
- [ ] An `MCDropoutPredictor` built from only the MC-Dropout path needs no deterministic model and exposes no `predict`/`explain` methods.
- [ ] mypy clean; pytest green.

### Technical notes

- `MCDropoutPredictor.from_path(model_path)` loads the MCD model as its single session (inherited from `BasePredictor.from_path` — the MCD model is just "the model" for this capability). `predict_with_uncertainty` preprocesses via `preprocess_image` (#2) then delegates to `mc_dropout_predict(self_session, arr, n_passes)`.
- Port `mc_dropout_predict` aggregation verbatim from legacy `predictor.py:364-406` (softmax per pass, mean/std stack, entropy with `1e-12` epsilon).
- The legacy `Predictor.predict_with_uncertainty` raised `RuntimeError` when no MCD session was supplied to a combined predictor. That coupling disappears here: `MCDropoutPredictor` always owns its MCD session, so the "no mcd session" error path no longer exists by construction. The legacy behavior stays covered by `test_mc_dropout.py` against old `Predictor` until #9.
- `test_mc_dropout.py` drives the same observable aggregation contract; it must hold through `MCDropoutPredictor.predict_with_uncertainty`.
