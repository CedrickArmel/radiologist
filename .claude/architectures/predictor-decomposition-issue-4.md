## ✨ MCDropoutPredictor, smart create_app factory, and 3-command CLI

### Context

Final slice. Replaces the skeleton stubs for `MCDropoutPredictor.predict_with_uncertainty`, the `create_app` smart factory, and reworks the CLI. These are bundled (pragmatic-balance: the factory must dispatch over all three concrete classes, so it lands once they exist, and the CLI shares the same wiring — splitting them would be needless churn and an extra round-trip). Reaches GREEN-real for uncertainty, serving, and CLI. Requires #2 and #3 (the factory dispatches over `Classifier`, `Explainer`, `MCDropoutPredictor`). See `predictor-decomposition-spec.md`.

### User story

As an **operator**, I want to **serve or invoke exactly the capability a given model supports** so that **each capability can run as its own independently containerized service or CLI command**.

### Acceptance criteria

**MCDropoutPredictor**

- [ ] Given an MC-Dropout ONNX model and an image, `MCDropoutPredictor.from_path(model_path).predict_with_uncertainty(image, n_passes=N)` returns mean probabilities and per-class std (both keyed by class name), a non-negative `predictive_entropy`, and `n_passes == N`.
- [ ] Across repeated calls the stochastic model yields a non-zero per-class std for at least one class (passes are genuinely independent).

**Smart create_app factory**

- [ ] `create_app(Classifier(...))` serves `POST /predict` and `GET /healthz`, and returns 404 for `POST /explain` and `POST /uncertainty`.
- [ ] `create_app(Explainer(...))` serves `POST /predict`, `POST /explain`, and `GET /healthz`, and returns 404 for `POST /uncertainty`.
- [ ] `create_app(MCDropoutPredictor(...))` serves `POST /uncertainty` and `GET /healthz`, and returns 404 for `POST /predict` and `POST /explain`.
- [ ] Any wired inference route returns 400 for an empty or unidentifiable image upload.
- [ ] `GET /healthz` returns 200 with an ok status when a predictor is loaded and 503 when none is.
- [ ] When fastapi is unavailable, `create_app` raises `RuntimeError` whose message names the `serve` extra.

**CLI**

- [ ] `predict <image> --model <path>` prints the predicted class and per-class probabilities and exits 0; on any load/inference error it prints an error to stderr and exits non-zero.
- [ ] `explain <image> --model <path>` runs Score-CAM and exits 0 (writing/printing the saliency result); errors exit non-zero.
- [ ] `uncertainty <image> --model <path>` runs MC-Dropout and prints entropy and per-class std, exiting 0; errors exit non-zero.
- [ ] The CLI exposes no `pull` command.

- [ ] mypy clean; pytest green.

### Design notes

The factory dispatches by concrete type using subclass checks ordered most-specific first (`Explainer` before `Classifier`, since `Explainer` is-a `Classifier`): an `Explainer` gets both `/predict` and `/explain`; a plain `Classifier` gets only `/predict`. This keeps route wiring a single function in `_build_app` with no per-class app files — capability is read off the object the caller already constructed, so a caller who built a `Classifier` cannot accidentally expose explanation. `/healthz` is always registered.

### Technical notes

- `_build_app(fastapi_mod, predictor)` in `app.py` is extended to register routes conditionally on `isinstance(predictor, ...)`. Keep the existing `_load_pil` / `_get_predictor` / validation-handler helpers.
- `cli.py`: add `explain` and `uncertainty` Typer commands mirroring `predict`; delete the `pull` command and its `pull_model` import.
- `test_app.py` is reworked to assert per-subclass route availability (404 on unsupported routes); `test_cli.py` is reworked to cover `predict`/`explain`/`uncertainty` and to assert `pull` is gone; `test_mc_dropout.py` is re-pointed to `MCDropoutPredictor`.
- Mock only true boundaries (image upload bytes, missing fastapi). Build real `Classifier`/`Explainer`/`MCDropoutPredictor` instances from the existing ONNX test fixtures rather than mocking predictors.
