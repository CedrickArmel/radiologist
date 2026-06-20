## ✨ Implement `MCDropoutPredictor`

### Context

This slice replaces the skeleton stubs for the uncertainty path:
`MCDropoutPredictor.predict_with_uncertainty` and the stateless `mc_dropout_predict`
helper. `MCDropoutPredictor` subclasses `BasePredictor` directly (not `Classifier`) — it
needs only the shared constructors, metadata, and preprocessing, plus a stochastic
session. Implement top-down to GREEN-real: no `NotImplementedError` reachable through
`MCDropoutPredictor.from_path(...).predict_with_uncertainty(...)`. See the epic spec for
architecture. This issue depends only on the skeleton and may run in parallel with #2.

**Blocked by:** #1 (skeleton).

### User story

As a **caller needing calibrated confidence**, I want to **run repeated stochastic forward
passes and get mean probabilities, per-class spread, and predictive entropy** so that **I
can flag low-confidence predictions for human review**.

### What to implement

**`mc_dropout.py`**:

- `mc_dropout_predict(session, image, n_passes=30)` — port verbatim from the monolith:
  read classes from session metadata, run `n_passes` stochastic forward passes for
  `"logits"`, softmax each, stack, compute mean / std / predictive entropy
  (`-Σ mean·log(mean+1e-12)`), return `UncertaintyResult`. Stays public and re-exported
  from `__init__.py` (wired in #1).
- `MCDropoutPredictor.predict_with_uncertainty(image, n_passes=30)`:
  - require a stochastic (mcd) session loaded via `from_path(det_path, mcd_path=...)`;
    when absent, raise `RuntimeError` naming MC-Dropout;
  - read `input_shape` from metadata, preprocess the image via the inherited helper,
    delegate to `mc_dropout_predict(mcd_session, arr, n_passes)`.

### Tests

Own the uncertainty behavioral tests (migrate `test_mc_dropout.py`). Drive through the
public API using the real stochastic fixture (`build_mcd_onnx`, `mcd_onnx_path`) and a det
fixture for the constructor.

- Given a det+mcd model, `MCDropoutPredictor.from_path(det_path, mcd_path=...)` then
  `predict_with_uncertainty(image, n_passes=k)` returns an `UncertaintyResult` with
  `n_passes == k`, `mean_probabilities`/`std_per_class` keyed by the model classes, and a
  finite non-negative `predictive_entropy`.
- Because the fixture session is stochastic, `std_per_class` contains at least one
  non-zero value across passes (the spread is observable).
- When the predictor was loaded with no `mcd_path`, `predict_with_uncertainty` raises
  `RuntimeError` naming MC-Dropout.
- `mc_dropout_predict(session, image, n_passes=k)` called directly on a stochastic session
  returns an `UncertaintyResult` with the same shape contract (covers the public helper).

### Acceptance criteria

- [ ] Given a det+mcd model, `predict_with_uncertainty(image, n_passes=k)` returns an `UncertaintyResult` with `n_passes == k` and class-keyed mean/std maps.
- [ ] `predictive_entropy` is finite and non-negative; `std_per_class` shows observable spread for the stochastic session.
- [ ] When loaded without an mcd session, `predict_with_uncertainty` raises `RuntimeError` naming MC-Dropout.
- [ ] `mc_dropout_predict` called directly on a stochastic session returns an `UncertaintyResult` with the documented fields.
- [ ] mypy clean; pytest green.
