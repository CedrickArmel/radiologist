## ✨ Smart `create_app` + per-capability routers

### Context

Replaces the monolithic `_build_app` (one app carrying all three routes) with three per-capability router builders and a smart `create_app` factory that inspects the injected predictor's concrete type and composes **only** the matching router plus `/healthz`. This makes each capability independently servable and containerizable: a `Classifier` app exposes only `/predict`, an `Explainer` app exposes `/predict` + `/explain`, an `MCDropoutPredictor` app exposes only `/uncertainty`. Replaces the router skeleton stubs and rewrites `create_app`. Requires: #3, #4, #5 (the capability methods the routers call must be GREEN-real). Target GREEN-real: no `NotImplementedError` reachable through `create_app(<capability>)`.

### User story

As a **platform operator**, I want each capability to serve only its own endpoints so that I can deploy and scale classification, explanation, and uncertainty as separate containers.

### Acceptance criteria

- [ ] `create_app(classifier)` serves `POST /predict` (200 with probabilities) and `GET /healthz`, and returns 404 for `/explain` and `/uncertainty`.
- [ ] `create_app(explainer)` serves `POST /predict`, `POST /explain` (200 with a saliency map), and `GET /healthz`; it returns 404 for `/uncertainty`.
- [ ] `create_app(mc_dropout_predictor)` serves `POST /uncertainty` (200 with mean probabilities and entropy) and `GET /healthz`, and returns 404 for `/predict`.
- [ ] For every mounted inference route, posting an empty or non-image file returns 400.
- [ ] For every mounted inference route, when the app was created with no predictor, the route returns 503.
- [ ] `GET /healthz` returns 200 when a predictor is loaded.
- [ ] When the `serve` extra (fastapi) is not installed, `create_app` raises `RuntimeError` naming the `serve` extra.
- [ ] mypy clean; pytest green.

### Interface contracts

##### `radiologist-inference/src/radiologist/inference/app.py`

```python
def create_app(predictor: Optional[BasePredictor] = None) -> Any:
    # contract: raises RuntimeError naming 'serve' if fastapi absent; otherwise builds a FastAPI
    #           app, mounts /healthz, and includes exactly the router(s) matching the predictor's
    #           concrete capability type (Explainer => classifier+explainer routers).
```

##### `radiologist-inference/src/radiologist/inference/classifier_router.py`

```python
def build_classifier_router(fastapi_mod: Any, predictor: Optional[Classifier]) -> Any:
    # contract: APIRouter with POST /predict -> {probabilities, predicted_class};
    #           400 on empty/invalid image, 503 when predictor is None.
```

##### `radiologist-inference/src/radiologist/inference/explainer_router.py`

```python
def build_explainer_router(fastapi_mod: Any, predictor: Optional[Explainer]) -> Any:
    # contract: APIRouter with POST /explain -> {saliency_map, predicted_class};
    #           400 on empty/invalid image, 503 when predictor is None.
```

##### `radiologist-inference/src/radiologist/inference/mc_dropout_router.py`

```python
def build_mc_dropout_router(fastapi_mod: Any, predictor: Optional[MCDropoutPredictor]) -> Any:
    # contract: APIRouter with POST /uncertainty -> {mean_probabilities, std_per_class,
    #           predictive_entropy, n_passes}; 400 on empty/invalid image, 503 when predictor None.
```

### Technical notes

- Each router builder owns its own shared `_load_pil` / `_get_predictor(503)` / empty-file(400) helpers and the `RequestValidationError -> 400` handler registration, ported from legacy `app.py:54-128`. Factor the duplicated request-handling helpers into a small private module reached only through the routers if duplication bothers the implementer (optional; not required for GREEN).
- Smart dispatch in `create_app`: detect type with `isinstance`. Because `Explainer` is a `Classifier`, an `Explainer` predictor mounts **both** the classifier and explainer routers — check most-specific first (`Explainer` before `Classifier`) or include the classifier router for any `Classifier` instance plus the explainer router additionally when `isinstance(predictor, Explainer)`.
- `MCDropoutPredictor` is not a `Classifier`, so its app mounts only the uncertainty router.
- `test_app.py` currently asserts all three routes on one app; this slice changes that contract — the route set is now capability-scoped. Update those tests to the per-capability contract above (this is intended behavior change, allowed because the route surface is the public behavior being redesigned).
- Use `fastapi.APIRouter` and `app.include_router(...)`. Keep the `serve`-extra `RuntimeError` guard from legacy `create_app`.

### Design notes

`create_app` is a smart factory keyed on the injected instance's type rather than a flag argument: the predictor already encodes which capabilities exist, so a separate "which routes?" parameter would be a redundant, drift-prone second source of truth. One router builder per capability means each endpoint has exactly one reason to change and can be composed independently — the core of the independent-servability goal.
