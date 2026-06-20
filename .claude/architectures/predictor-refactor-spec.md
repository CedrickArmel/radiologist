# 🚀 Epic — Decompose `Predictor` into capability subclasses (clean architecture)

## Problem Statement

The single `Predictor` class fuses three unrelated capabilities (classification, Score-CAM explanation, MC-Dropout uncertainty), loads every ONNX model whether or not a capability is used, and forces one FastAPI app + one CLI to carry all three concerns — so no capability can be loaded, served, or containerized independently, and every concern shares one reason to change.

## Goal

A caller can construct exactly the capability they need (`Classifier`, `Explainer`, or `MCDropoutPredictor`), each loading only its own ONNX model(s), serve any one of them through `create_app(predictor)` which auto-wires only the matching routes, and drive each from its own CLI subcommand — with each capability independently testable, servable, and containerizable.

## Scope

**In scope:**

- `BasePredictor` (shared loading + preprocessing seam) + `Classifier(BasePredictor)` + `Explainer(Classifier)` + `MCDropoutPredictor(BasePredictor)`.
- One file per concern: `base.py`, `classifier.py`, `explainer.py`, `mc_dropout.py`, `preprocessing.py`, `results.py`, plus per-capability routers `classifier_router.py`, `explainer_router.py`, `mc_dropout_router.py` composed by a smart `create_app`.
- `create_app(predictor)` auto-detects the concrete subclass and mounts only its routes (+ `/healthz`).
- CLI: three subcommands `predict`, `explain`, `uncertainty`. `pull` removed (owned by `radiologist-registry`).
- `from_registry()` delegates to `radiologist.registry.WandbRegistry().pull(...)`; the `pull_model` bridge is removed.
- Breaking `__all__` change: `Predictor` and `pull_model` removed; capability classes added.

**Out of scope:**

- Any change to Score-CAM math (`cam.py`) or MC-Dropout aggregation math — only their call sites move.
- The `radiologist-registry` package itself (epic #85–90) — this epic consumes it.
- Multi-model / ensemble serving in one app — one capability per app is the design.

## Architecture summary

Clean-architecture decomposition built around a **template-method base + capability subclasses + composable routers**. `BasePredictor` owns the deterministic-session lifecycle (`from_path`, `from_registry`, metadata parsing) and the preprocessing seam; it is an **ABC**, not a Protocol, because subclasses genuinely share loading/preprocessing implementation (a Protocol would force every subclass to re-implement it, defeating the reuse goal). `Classifier` adds `predict`; `Explainer(Classifier)` adds `explain` (reusing classify for the predicted label); `MCDropoutPredictor(BasePredictor)` loads only the MCD session and adds `predict_with_uncertainty`. Stateless math (`score_cam`, `mc_dropout_predict`) and preprocessing helpers live in their own modules and are reached only through the public capability methods — never mocked in tests. The serving layer splits into one router builder per capability; `create_app` is a smart factory that inspects the injected predictor's type and composes exactly the matching router plus `/healthz`. Result dataclasses move to `results.py` and become **frozen** (immutable value objects — they are pure outputs with no reason to mutate). Per-issue contracts live in the issue files.

## Acceptance criteria

- [ ] A caller can build a `Classifier` from a single deterministic ONNX path and get per-class probabilities, with no MC-Dropout model loaded.
- [ ] A caller can build an `Explainer` from a deterministic ONNX path and get both a saliency map and the predicted class.
- [ ] A caller can build an `MCDropoutPredictor` from an MC-Dropout ONNX path and get mean probabilities, per-class spread, and predictive entropy, with no deterministic explanation model required.
- [ ] `create_app(classifier)` serves `POST /predict` + `GET /healthz` only; `create_app(explainer)` serves `/predict` + `/explain` + `/healthz`; `create_app(mc_dropout_predictor)` serves `/uncertainty` + `/healthz`.
- [ ] Each capability is constructible from the W&B registry via `from_registry`, delegating to `WandbRegistry().pull`.
- [ ] The CLI exposes `predict`, `explain`, and `uncertainty` subcommands and no `pull` subcommand.
- [ ] Importing `radiologist.inference` no longer exposes `Predictor` or `pull_model`; it exposes `BasePredictor`, `Classifier`, `Explainer`, `MCDropoutPredictor`.
- [ ] `score_cam` and `mc_dropout_predict` remain importable from `radiologist.inference`.
- [ ] mypy clean; pytest green.

## Public API contract (authoritative)

This is the complete public surface the skeleton (Issue #1) stubs. All bodies `raise NotImplementedError`.

```python
# radiologist/inference/results.py
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

```python
# radiologist/inference/base.py
class BasePredictor(abc.ABC):
    """Shared ONNX session lifecycle + metadata for every capability."""
    @classmethod
    def from_path(cls, model_path: str) -> "BasePredictor": ...
    @classmethod
    def from_registry(cls, artifact_path: str, local_dir: str) -> "BasePredictor": ...
    @property
    def metadata(self) -> ModelMetadata: ...
```

```python
# radiologist/inference/classifier.py
class Classifier(BasePredictor):
    def predict(
        self,
        image: Union[str, np.ndarray, "PILImage.Image"],
        deployment_prior: Optional[Dict[str, float]] = None,
    ) -> Prediction: ...
```

```python
# radiologist/inference/explainer.py
class Explainer(Classifier):
    def explain(
        self,
        image: Union[str, np.ndarray, "PILImage.Image"],
    ) -> Explanation: ...
```

```python
# radiologist/inference/mc_dropout.py
class MCDropoutPredictor(BasePredictor):
    def predict_with_uncertainty(
        self,
        image: Union[str, np.ndarray, "PILImage.Image"],
        n_passes: int = 30,
    ) -> UncertaintyResult: ...
```

```python
# radiologist/inference/preprocessing.py
def preprocess_image(
    image: Union[str, np.ndarray, "PILImage.Image"],
    input_shape: List[int],
) -> np.ndarray: ...
def read_metadata(session: "ort.InferenceSession") -> Dict[str, str]: ...
```

```python
# radiologist/inference/cam.py  (unchanged public re-export)
def score_cam(feature_maps: np.ndarray, logits: np.ndarray) -> np.ndarray: ...

# radiologist/inference/mc_dropout.py
def mc_dropout_predict(session: Any, image: np.ndarray, n_passes: int = 30) -> UncertaintyResult: ...
```

```python
# radiologist/inference/app.py
def create_app(predictor: Optional["BasePredictor"] = None) -> Any: ...
```

```python
# __all__ (radiologist/inference/__init__.py)
__all__ = [
    "BasePredictor", "Classifier", "Explainer", "MCDropoutPredictor",
    "score_cam", "mc_dropout_predict",
    "Prediction", "Explanation", "UncertaintyResult", "ModelMetadata",
    "create_app",
]
```

## Build sequence

| Local # | Title                                                    | File                              | Depends on | Phase |
| ------- | -------------------------------------------------------- | --------------------------------- | ---------- | ----- |
| 1       | Inference capability split — skeleton                    | predictor-refactor-issue-1.md     | —          | 1     |
| 2       | Preprocessing + metadata seam                            | predictor-refactor-issue-2.md     | 1          | 2     |
| 3       | `Classifier.predict`                                     | predictor-refactor-issue-3.md     | 1, 2       | 3     |
| 4       | `Explainer.explain`                                      | predictor-refactor-issue-4.md     | 1, 2, 3    | 3     |
| 5       | `MCDropoutPredictor.predict_with_uncertainty`            | predictor-refactor-issue-5.md     | 1, 2       | 3     |
| 6       | `from_registry` delegation to `WandbRegistry`            | predictor-refactor-issue-6.md     | 1, 2       | 3     |
| 7       | Smart `create_app` + per-capability routers              | predictor-refactor-issue-7.md     | 3, 4, 5    | 4     |
| 8       | Three-subcommand CLI (`predict`/`explain`/`uncertainty`) | predictor-refactor-issue-8.md     | 3, 4, 5    | 4     |
| 9       | Cleanup — remove `Predictor`/`pull_model`, prune `__all__` | predictor-refactor-issue-9.md   | 7, 8       | 5     |

Epic shape: **1 skeleton (#1) + 7 outside-in slices (#2–#8) + 1 cleanup (#9)**. The whole epic sequences **after** registry epic #85–90 is merged (it imports `radiologist.registry.WandbRegistry`).

## Dependencies

- **Registry epic #85–90** (`radiologist-registry`) — must be merged first. `BasePredictor.from_registry` imports `WandbRegistry` from `radiologist.registry`. Registry issue #5 already removes `pull_model` and rewires `from_registry`; this epic supersedes that wiring against the new class hierarchy — coordinate so #9 here is the final authority on `from_registry`.
- External packages (unchanged): `onnxruntime`, `numpy`, `Pillow`; optional extras `serve` (fastapi), `cli` (typer), `registry` (wandb).
