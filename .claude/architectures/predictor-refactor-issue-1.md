## 🦴 Inference capability split — skeleton

### Context

Stubs the entire decomposed inference surface so every slice (#2–#8) can start in parallel, blocked only on this. Creates the new per-concern modules (`base.py`, `classifier.py`, `explainer.py`, `mc_dropout.py`, `preprocessing.py`, `results.py`, and the three router modules) and the class hierarchy as type-checked contracts only. **No behavior** — every method body is `raise NotImplementedError`. The legacy `predictor.py` symbols (`Predictor`, `pull_model`) stay in place and in `__all__` until cleanup (#9), so the existing test suite stays green throughout. See the epic spec (`predictor-refactor-spec.md`) for the architecture rationale. The epic sequences after registry epic #85–90.

### Module layout

```
radiologist-inference/src/radiologist/inference/
├── __init__.py            # exports updated to add new classes; legacy names kept until #9
├── results.py             # frozen result dataclasses (moved out of predictor.py)
├── base.py                # BasePredictor ABC — session lifecycle + metadata
├── classifier.py          # Classifier(BasePredictor) — predict
├── explainer.py           # Explainer(Classifier) — explain
├── mc_dropout.py          # MCDropoutPredictor + mc_dropout_predict helper
├── preprocessing.py       # preprocess_image, read_metadata (moved out of predictor.py)
├── classifier_router.py   # build_classifier_router(fastapi, predictor)
├── explainer_router.py    # build_explainer_router(fastapi, predictor)
├── mc_dropout_router.py   # build_mc_dropout_router(fastapi, predictor)
├── app.py                 # create_app smart factory (currently legacy _build_app)
├── cam.py                 # unchanged
├── cli.py                 # unchanged until #8
├── optional.py            # unchanged
└── predictor.py           # legacy Predictor + pull_model kept until #9
```

### Interface contracts

All bodies `raise NotImplementedError`. Full type hints, no logic.

##### `radiologist-inference/src/radiologist/inference/results.py`

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

##### `radiologist-inference/src/radiologist/inference/preprocessing.py`

```python
def preprocess_image(
    image: Union[str, np.ndarray, PILImage.Image],
    input_shape: List[int],
) -> np.ndarray:
    # contract: load/resize/normalize to float32 (1, C, H, W) in [0, 1]; accepts path, HWC uint8, or PIL
    raise NotImplementedError

def read_metadata(session: ort.InferenceSession) -> Dict[str, str]:
    # contract: return dict(session.get_modelmeta().custom_metadata_map)
    raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/base.py`

```python
class BasePredictor(abc.ABC):
    # contract: owns the ONNX session + parsed metadata; subclasses add capability methods.
    @classmethod
    def from_path(cls, model_path: str) -> "BasePredictor":
        # contract: open InferenceSession(model_path), parse metadata; raises FileNotFoundError
        #           if path absent, InvalidGraph if not a valid ONNX model.
        raise NotImplementedError

    @classmethod
    def from_registry(cls, artifact_path: str, local_dir: str) -> "BasePredictor":
        # contract: WandbRegistry().pull(artifact_path, local_dir) -> path, then cls.from_path(path);
        #           raises RuntimeError when the 'registry' extra (wandb) is absent.
        raise NotImplementedError

    @property
    def metadata(self) -> ModelMetadata:
        # contract: typed view over the embedded ONNX metadata
        raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/classifier.py`

```python
class Classifier(BasePredictor):
    def predict(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
        deployment_prior: Optional[Dict[str, float]] = None,
    ) -> Prediction:
        # contract: deterministic softmax; deployment_prior overrides embedded training_prior;
        #           neither present -> raw softmax. Returns probabilities + argmax label.
        raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/explainer.py`

```python
class Explainer(Classifier):
    def explain(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
    ) -> Explanation:
        # contract: Score-CAM saliency map sized to the original image + predicted class label.
        raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/mc_dropout.py`

```python
class MCDropoutPredictor(BasePredictor):
    def predict_with_uncertainty(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
        n_passes: int = 30,
    ) -> UncertaintyResult:
        # contract: n_passes stochastic forward passes -> mean probs, per-class std, entropy.
        raise NotImplementedError

def mc_dropout_predict(
    session: Any,
    image: np.ndarray,
    n_passes: int = 30,
) -> UncertaintyResult:
    # contract: stateless aggregation over n_passes stochastic runs of a preprocessed array.
    raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/classifier_router.py`

```python
def build_classifier_router(fastapi_mod: Any, predictor: Optional[Any]) -> Any:
    # contract: APIRouter exposing POST /predict bound to predictor.predict
    raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/explainer_router.py`

```python
def build_explainer_router(fastapi_mod: Any, predictor: Optional[Any]) -> Any:
    # contract: APIRouter exposing POST /explain bound to predictor.explain
    raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/mc_dropout_router.py`

```python
def build_mc_dropout_router(fastapi_mod: Any, predictor: Optional[Any]) -> Any:
    # contract: APIRouter exposing POST /uncertainty bound to predictor.predict_with_uncertainty
    raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/__init__.py`

```python
__all__ = [
    "BasePredictor", "Classifier", "Explainer", "MCDropoutPredictor",
    # legacy, removed in #9:
    "Predictor", "pull_model",
    "score_cam", "mc_dropout_predict",
    "Prediction", "Explanation", "UncertaintyResult", "ModelMetadata",
    "create_app",
]
```

### Technical notes

- `score_cam` and `mc_dropout_predict` must stay importable from the package (the existing `test_public_api.py` asserts on the legacy `__all__`; this issue keeps the legacy names so that test stays green — the cleanup #9 updates it).
- `results.py` dataclasses are `frozen=True`. `np.ndarray` in a frozen dataclass is fine — frozen blocks attribute rebinding, not array mutation; that is acceptable for a value object.
- Each new file lives under `src/radiologist/inference/`; do not add a license header (the `insert-license` pre-commit hook does this).
- Python 3.10: use `Optional[...]` / `Union[...]` from `typing`, not `X | Y`.

### Acceptance criteria

- [ ] `from radiologist.inference import BasePredictor, Classifier, Explainer, MCDropoutPredictor` succeeds.
- [ ] mypy clean; pytest green (no behavioral tests — stubs typecheck and the existing suite stays green).
