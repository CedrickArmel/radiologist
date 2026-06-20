# 🚀 Epic — Decompose `Predictor` into a one-file-per-class hierarchy

## Problem Statement

`radiologist.inference.predictor` is a single monolithic module that bundles four
unrelated responsibilities — deterministic classification, Score-CAM explanation,
MC-Dropout uncertainty, and W&B model pulling — behind one fat `Predictor` class. A
caller that only wants classification is forced to depend on the entire surface, and the
single class cannot express which capabilities a loaded model actually supports.

## Goal

Replace the monolith with a small inheritance hierarchy — one class per file —
(`BasePredictor` → `Classifier` → `Explainer`, plus `MCDropoutPredictor`) wired into a
single smart `create_app` factory and three focused CLI subcommands, with no behavioral
regression in prediction, explanation, or uncertainty output.

## Scope

**In scope:**
- New modules: `models.py`, `base_predictor.py`, `classifier.py`, `explainer.py`, `mc_dropout.py`.
- Refactor `app.py` (`create_app` becomes an `isinstance`-driven factory) and `cli.py` (three subcommands).
- `from_registry()` delegates to `WandbRegistry().pull()` from `radiologist-registry`, with an injectable `registry` parameter.
- Remove `Predictor` and `pull_model` from the public API; delete `predictor.py`.

**Out of scope:**
- Any change to `cam.py` (`score_cam`, `score_cam_with_session`) or `optional.py`.
- The W&B pull implementation itself — owned by `radiologist-registry` (issues #85–90).
- A `pull` CLI subcommand — owned by the registry package.
- New inference behaviors (batching, new model formats, new CAM variants).

## Architecture summary

Chosen approach: **one class per file behind a shared abstract base**. `BasePredictor`
owns ONNX session loading, metadata reading, image preprocessing, prior correction, and
the two constructors (`from_path`, `from_registry`); it exposes no inference verb.
`Classifier(BasePredictor)` adds `predict`. `Explainer(Classifier)` adds `explain` and
inherits `predict`. `MCDropoutPredictor(BasePredictor)` adds `predict_with_uncertainty`
and requires a stochastic session. `create_app(predictor)` inspects the concrete subclass
via `isinstance` and wires only the routes the instance can serve, returning 404-free
apps with exactly the supported endpoints plus `/healthz`. The result dataclasses move
unchanged to `models.py`. `score_cam` and `mc_dropout_predict` remain stateless public
helpers re-exported from the package. See the issue files for per-issue interface
contracts and acceptance criteria.

## Public API contract (post-epic `__all__`)

The package `__init__.py` re-exports exactly the following. `Predictor` and `pull_model`
are **removed** (breaking change).

```python
__all__ = [
    "BasePredictor",
    "Classifier",
    "Explainer",
    "MCDropoutPredictor",
    "Prediction",
    "Explanation",
    "UncertaintyResult",
    "ModelMetadata",
    "score_cam",
    "mc_dropout_predict",
    "create_app",
]
```

### Result dataclasses — `models.py` (frozen)

```python
@dataclass(frozen=True)
class Prediction:
    probabilities: Dict[str, float]
    predicted_class: str

@dataclass(frozen=True)
class Explanation:
    saliency_map: np.ndarray
    predicted_class: str

@dataclass(frozen=True)
class UncertaintyResult:
    mean_probabilities: Dict[str, float]
    std_per_class: Dict[str, float]
    predictive_entropy: float
    n_passes: int

@dataclass(frozen=True)
class ModelMetadata:
    classes: List[str]
    input_shape: List[int]
    cam_target_layer: str
    output_names: List[str]
    mc_dropout: bool
```

### `BasePredictor` — `base_predictor.py`

```python
class BasePredictor:
    # Shared ONNX-backed base. Loads the session, reads metadata, and provides
    # preprocessing + prior correction to subclasses. Exposes no inference verb.

    @classmethod
    def from_path(
        cls, det_path: str, mcd_path: Optional[str] = None
    ) -> "Self":
        # contract: opens det_path as an InferenceSession, reads custom metadata,
        # optionally opens mcd_path; returns an instance of the calling subclass.

    @classmethod
    def from_registry(
        cls,
        artifact_path: str,
        local_dir: str,
        registry: Optional["ModelRegistry"] = None,
    ) -> "Self":
        # contract: when registry is None, instantiates WandbRegistry();
        # calls registry.pull(artifact_path, local_dir) to obtain a local det path,
        # then delegates to from_path. Raises RuntimeError naming 'registry' when the
        # registry extra (wandb) is unavailable.
```

Module-private helpers (not exported, exercised through subclass public APIs):
`_read_metadata(session) -> Dict[str, str]`,
`_preprocess_image(image, input_shape) -> np.ndarray`,
`_apply_prior_correction(softmax, classes, prior) -> np.ndarray`.

### `Classifier(BasePredictor)` — `classifier.py`

```python
class Classifier(BasePredictor):
    def predict(
        self,
        image: Union[str, np.ndarray, "PILImage.Image"],
        deployment_prior: Optional[Dict[str, float]] = None,
    ) -> Prediction:
        # contract: preprocess image, run det session for "logits", softmax,
        # apply deployment_prior or embedded training_prior when present,
        # return Prediction(probabilities, predicted_class=argmax).
```

### `Explainer(Classifier)` — `explainer.py`

```python
class Explainer(Classifier):
    def explain(
        self, image: Union[str, np.ndarray, "PILImage.Image"]
    ) -> Explanation:
        # contract: run det session for "logits" + "feature_maps", compute Score-CAM
        # saliency at the original image resolution, return
        # Explanation(saliency_map, predicted_class). Inherits predict from Classifier.
```

### `MCDropoutPredictor(BasePredictor)` — `mc_dropout.py`

```python
class MCDropoutPredictor(BasePredictor):
    def predict_with_uncertainty(
        self,
        image: Union[str, np.ndarray, "PILImage.Image"],
        n_passes: int = 30,
    ) -> UncertaintyResult:
        # contract: requires a stochastic (mcd) session; raises RuntimeError naming
        # MC-Dropout when none was loaded. Runs n_passes stochastic forward passes,
        # returns UncertaintyResult(mean, std, predictive_entropy, n_passes).
```

### Stateless helpers (unchanged behavior, re-exported)

```python
def score_cam(feature_maps: np.ndarray, logits: np.ndarray) -> np.ndarray: ...
def mc_dropout_predict(
    session: ort.InferenceSession, image: np.ndarray, n_passes: int = 30
) -> UncertaintyResult: ...
```

### Serving factory — `app.py`

```python
def create_app(predictor: Optional["BasePredictor"] = None) -> Any:
    # contract: raises RuntimeError naming 'serve' when fastapi is unavailable.
    # Inspects predictor via isinstance and wires only matching routes:
    #   Classifier            -> POST /predict
    #   Explainer             -> POST /predict, POST /explain
    #   MCDropoutPredictor    -> POST /uncertainty
    # Always wires GET /healthz. Routes the instance cannot serve are absent (404).
```

## Build sequence

| Phase | Local # | Issue title                                   | Files changed                                   | Depends on |
| ----- | ------- | --------------------------------------------- | ----------------------------------------------- | ---------- |
| 1     | 1       | Skeleton — module split + typed contracts     | `models.py`, `base_predictor.py`, `classifier.py`, `explainer.py`, `mc_dropout.py`, `app.py`, `cli.py`, `__init__.py`, `tests/test_public_api.py` | Registry #85–90 |
| 2     | 2       | Implement `Classifier` + `BasePredictor` core | `base_predictor.py`, `classifier.py`            | #1         |
| 2     | 4       | Implement `MCDropoutPredictor`                | `mc_dropout.py`                                 | #1         |
| 3     | 3       | Implement `Explainer`                         | `explainer.py`                                  | #1, #2     |
| 4     | 5       | Refactor `app.py` smart factory + `cli.py`    | `app.py`, `cli.py`                              | #1, #2, #3, #4 |
| 5     | 6       | Cleanup — delete `predictor.py`, finalize API | `predictor.py` (deleted), `__init__.py`, `tests/test_public_api.py`, `tests/test_registry.py` | #2, #3, #4, #5 |

## Dependency graph

- **Epic** depends on the registry epic (issues #85–90) being merged: `from_registry`
  imports `WandbRegistry` / `ModelRegistry` from `radiologist-registry`.
- **#1 (skeleton)** blocks every other issue. It stubs all five new modules, rewires
  `app.py`/`cli.py` imports against the stubs, updates `__init__.py` and
  `tests/test_public_api.py`, and leaves the suite green with no behavioral tests.
- **#2** and **#4** depend only on #1 and may proceed in parallel.
- **#3** depends on #1 and **#2** (it subclasses `Classifier` and inherits `predict`).
- **#5** depends on #1, #2, #3, #4 (the factory and CLI must wire all concrete classes).
- **#6** depends on #2, #3, #4, #5 — it removes the monolith only after every behavior
  has a real home and the suite is green.

## Acceptance criteria

- [ ] `import radiologist.inference` exposes exactly the post-epic `__all__`; `Predictor` and `pull_model` are no longer importable from the package.
- [ ] A `Classifier` loaded from an ONNX path returns a `Prediction` whose `probabilities` keys equal the model classes and whose `predicted_class` is the argmax.
- [ ] An `Explainer` returns an `Explanation` whose `saliency_map` matches the original image resolution and whose `predicted_class` agrees with `predict`.
- [ ] An `MCDropoutPredictor` returns an `UncertaintyResult` over the requested passes and raises `RuntimeError` (naming MC-Dropout) when no stochastic session was loaded.
- [ ] `create_app(Classifier(...))` serves `POST /predict` and `GET /healthz` and 404s on `/explain` and `/uncertainty`; `create_app(Explainer(...))` additionally serves `/explain`; `create_app(MCDropoutPredictor(...))` serves `/uncertainty`.
- [ ] The CLI exposes `predict`, `explain`, and `uncertainty` subcommands and no `pull` subcommand.
- [ ] `predictor.py` no longer exists; `mypy` clean; `pytest` green across the package.

## Dependencies

- **`radiologist-registry`** (issues #85–90) — provides `ModelRegistry` protocol and
  `WandbRegistry.pull(artifact_path, local_dir) -> str`. Must be merged before this epic
  starts.
- External extras unchanged: `registry` (wandb), `serve` (fastapi), `cli` (typer) — all
  guarded via the existing `optional.py` sentinels.
