## ✨ Smart `create_app` factory + three-command CLI

### Context

This slice replaces the skeleton stubs for `create_app` / `_build_app` and rewrites `cli.py`. The app factory becomes capability-aware: it inspects the injected predictor's concrete type and registers only the matching routes. The CLI swaps its two commands (`predict`, `pull`) for three (`predict`, `explain`, `uncertainty`), each constructing the matching subclass. This is the outermost slice — it composes the predictors from #2–#4, so it builds last among the slices. Drives real `_build_app` and a real FastAPI `TestClient` to GREEN-real. See `predictor-minimal-spec.md`.

> Requires: #1, #2, #3, #4.
> Blocks: —

### User story

As a **platform operator**, I want to **serve or invoke exactly one capability per deployment** so that **each capability is independently servable and containerizable from its own predictor instance**.

### Acceptance criteria

**Smart factory (`create_app`)**
- [ ] Given a `Classifier`, the app exposes `POST /predict` and `GET /healthz` and does not expose `/explain` or `/uncertainty` (those paths 404).
- [ ] Given an `Explainer`, the app exposes `POST /predict`, `POST /explain`, and `GET /healthz`, and does not expose `/uncertainty`.
- [ ] Given an `MCDropoutPredictor`, the app exposes `POST /uncertainty` and `GET /healthz` and does not expose `/predict` or `/explain`.
- [ ] `POST /predict` on a classifier-backed app returns 200 with a `probabilities` dict and a `predicted_class`.
- [ ] `POST /explain` on an explainer-backed app returns 200 with a nested-list `saliency_map` and a `predicted_class`.
- [ ] `POST /uncertainty` on an mc-dropout-backed app returns 200 with `std_per_class`, `mean_probabilities`, and a float `predictive_entropy`.
- [ ] On any exposed app, `GET /healthz` returns 200 with `{"status": "ok"}` when a predictor is loaded.
- [ ] When the uploaded image field is missing or its bytes are not a valid image, the relevant inference route returns 400.
- [ ] When `create_app(predictor=None)` is served, every exposed inference route returns 503.
- [ ] When the `serve` extra (fastapi) is absent, `create_app` raises `RuntimeError` whose message names the `serve` extra.

**CLI**
- [ ] `predict <image> --model <det>` prints the predicted class and exits 0; exits 1 when the image or model is unreadable.
- [ ] `explain <image> --model <det>` reports a saliency result and exits 0; exits 1 on unreadable input.
- [ ] `uncertainty <image> --model <mcd>` reports uncertainty and exits 0; exits 1 on unreadable input.
- [ ] No `pull` subcommand exists.
- [ ] When the `cli` extra (typer) is absent, invoking the CLI entry point raises `RuntimeError` whose message names the `cli` extra.
- [ ] mypy clean; pytest green.

### Data flow

```
create_app(predictor) → _build_app: isinstance(predictor, ...) → register {/predict?, /explain?, /uncertainty?} + /healthz
HTTP POST /predict → _load_pil(bytes) → classifier.predict(pil) → JSON
CLI predict → Classifier.from_path(model) → .predict(image) → echo
CLI explain → Explainer.from_path(model) → .explain(image) → echo
CLI uncertainty → MCDropoutPredictor.from_path(model) → .predict_with_uncertainty(image) → echo
```

### Technical notes

- `app.py` — keep the existing `_load_pil`, `_get_predictor` (503), and validation-error (400) helpers verbatim; gate each `@app.post` registration behind an `isinstance` check on the injected predictor. Order matters: check `Explainer`/`MCDropoutPredictor` capability via `isinstance`, and because `Explainer` is a `Classifier`, register `/predict` for any `Classifier` and `/explain` only for `Explainer`.
- `cli.py` — each command constructs the matching subclass via `from_path`; reuse the existing `try/except → typer.Exit(code=1)` error pattern. Remove the `pull` command and the `pull_model` import.
- Route handler bodies (read upload, `_load_pil`, call predictor, serialize) are reused verbatim from the current `app.py`.

### Design notes

Capability detection by `isinstance` (rather than a declared route list per predictor) keeps the factory a pure function of the existing type hierarchy — no new registry or capability-enum is introduced. This is the minimal-impact wiring: the smart factory is the only genuinely new behavior in serving, and it leans entirely on the subclass relationships defined in #2–#4.
