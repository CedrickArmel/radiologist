## ✨ MCDropoutPredictor — uncertainty estimation

### Context

This slice replaces the skeleton stub for `MCDropoutPredictor.predict_with_uncertainty`. `MCDropoutPredictor` extends `BasePredictor` (not `Classifier`) and loads a single stochastic ONNX model into the shared `_state.session`. It drives the real, unchanged `mc_dropout_predict` helper to GREEN-real. Independent of #2/#3 beyond the shared loading seam in #1, so it can run in parallel with them. See `predictor-minimal-spec.md`.

> Requires: #1.
> Blocks: #5 (app/CLI wire MCDropoutPredictor).

### User story

As a **risk-aware clinician**, I want to **estimate prediction uncertainty from a single MC-Dropout model** so that **I can stand up an uncertainty-only service that loads no deterministic classifier**.

### Acceptance criteria

- [ ] Given an MC-Dropout ONNX path, `MCDropoutPredictor.from_path(model_path)` returns an `MCDropoutPredictor` whose `predict_with_uncertainty(image)` yields an `UncertaintyResult` with one mean probability and one std per model class, a float `predictive_entropy`, and `n_passes` equal to the number of passes requested.
- [ ] Given `n_passes=k`, the result's `n_passes` equals `k` and the stochastic model is run `k` times.
- [ ] Given a stochastic model, repeated `predict_with_uncertainty` calls produce non-zero per-class std (the passes genuinely vary).
- [ ] Given an image as a file path, a NumPy HWC uint8 array, or a PIL Image, the result is well-formed (input form does not change the contract).
- [ ] mypy clean; pytest green.

### Technical notes

- `predictor.py` — reuse the old `predict_with_uncertainty` preprocessing + `mc_dropout_predict(self._state.session, arr, n_passes)` call; the single loaded model is now the MC-Dropout model itself, so the old "no mcd_session → RuntimeError" guard is dropped (a `MCDropoutPredictor` always holds an MC-Dropout model).
- `mc_dropout_predict(session, image, n_passes)` stays a public stateless helper — unchanged.

### Design notes

The old `Predictor` carried an optional second `mcd_session` and raised at call time when it was missing. Splitting MC-Dropout into its own subclass that loads the stochastic model as its only model removes that runtime guard entirely: the type now encodes the capability, which is the decomposition goal.
