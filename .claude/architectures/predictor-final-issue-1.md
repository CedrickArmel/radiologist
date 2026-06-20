## 🦴 Predictor decomposition — skeleton

### Context

This issue stubs the entire one-file-per-class surface so all slice issues (#2–#6) can
start blocked only on it. No inference behavior is implemented — the skeleton is a
type-checked contract, not a working system. It creates the five new modules as stubs,
rewires `app.py` and `cli.py` imports onto the new symbols, updates the package exports,
and rewrites the public-API test to assert the new surface. The existing behavioral test
files (prediction, explanation, uncertainty, registry, app, cli) still import the old
`Predictor`/`predictor` module and will be migrated by the slice issues that own them —
this issue must not break the *public-API* test or `mypy`.

**Blocked by:** the registry epic (#85–#90) must be merged — the `from_registry` stub
imports `ModelRegistry`/`WandbRegistry` from `radiologist-registry`.

### Module layout

```
radiologist-inference/src/radiologist/inference/
├── models.py           # Prediction, Explanation, UncertaintyResult, ModelMetadata (frozen)
├── base_predictor.py   # BasePredictor + _read_metadata, _preprocess_image, _apply_prior_correction
├── classifier.py       # Classifier(BasePredictor)
├── explainer.py        # Explainer(Classifier)
├── mc_dropout.py       # MCDropoutPredictor(BasePredictor) + mc_dropout_predict helper
├── app.py              # create_app smart factory (stubbed) + _build_app
├── cli.py              # Typer app: predict / explain / uncertainty (stubbed)
├── cam.py              # unchanged
├── optional.py         # unchanged
└── predictor.py        # still present; deleted in #6
```

### Interface contracts

All bodies are `raise NotImplementedError` except trivial re-exports and the extra-guard
sentinels (which must remain real so the extra-absent tests keep passing). Full type
hints. No inference logic.

##### `models.py`

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

##### `base_predictor.py`

```python
class BasePredictor:
    @classmethod
    def from_path(cls, det_path: str, mcd_path: Optional[str] = None) -> "BasePredictor":
        # contract: returns an instance of the calling subclass loaded from det_path
        raise NotImplementedError

    @classmethod
    def from_registry(
        cls,
        artifact_path: str,
        local_dir: str,
        registry: Optional["ModelRegistry"] = None,
    ) -> "BasePredictor":
        # contract: registry None -> WandbRegistry(); pull -> from_path.
        # raises RuntimeError naming 'registry' when the wandb extra is absent
        raise NotImplementedError


def _read_metadata(session: "ort.InferenceSession") -> Dict[str, str]:
    raise NotImplementedError


def _preprocess_image(image: Any, input_shape: List[int]) -> "np.ndarray":
    raise NotImplementedError


def _apply_prior_correction(
    softmax: "np.ndarray", classes: List[str], prior: Dict[str, float]
) -> "np.ndarray":
    raise NotImplementedError
```

> Use `from __future__ import annotations` (or `Optional["Self"]`-style string hints) so
> the `Self`/`ModelRegistry` forward references type-check under Python 3.10. The
> `ModelRegistry` import comes from `radiologist-registry`.

##### `classifier.py`

```python
class Classifier(BasePredictor):
    def predict(
        self,
        image: Union[str, "np.ndarray", "PILImage.Image"],
        deployment_prior: Optional[Dict[str, float]] = None,
    ) -> Prediction:
        raise NotImplementedError
```

##### `explainer.py`

```python
class Explainer(Classifier):
    def explain(
        self, image: Union[str, "np.ndarray", "PILImage.Image"]
    ) -> Explanation:
        raise NotImplementedError
```

##### `mc_dropout.py`

```python
class MCDropoutPredictor(BasePredictor):
    def predict_with_uncertainty(
        self,
        image: Union[str, "np.ndarray", "PILImage.Image"],
        n_passes: int = 30,
    ) -> UncertaintyResult:
        # contract: raises RuntimeError naming MC-Dropout when no stochastic session
        raise NotImplementedError


def mc_dropout_predict(
    session: "ort.InferenceSession", image: "np.ndarray", n_passes: int = 30
) -> UncertaintyResult:
    raise NotImplementedError
```

##### `app.py` — `create_app` stub (keep the real extra guard)

```python
def create_app(predictor: Optional["BasePredictor"] = None) -> Any:
    if _fastapi is None:
        raise RuntimeError("The 'serve' extra is required.")
    raise NotImplementedError
```

`_build_app` may stay as-is for now; #5 rewrites it into the smart factory.

##### `__init__.py` — new exports

```python
from radiologist.inference.app import create_app
from radiologist.inference.base_predictor import BasePredictor
from radiologist.inference.cam import score_cam
from radiologist.inference.classifier import Classifier
from radiologist.inference.explainer import Explainer
from radiologist.inference.mc_dropout import MCDropoutPredictor, mc_dropout_predict
from radiologist.inference.models import (
    Explanation,
    ModelMetadata,
    Prediction,
    UncertaintyResult,
)

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

> `score_cam` is re-exported from `cam.py` directly (the old monolith's thin wrapper is
> dropped). Confirm `cam.score_cam(feature_maps, logits)` matches the public signature; if
> the package historically re-exported a wrapper, point the import at `cam.score_cam`.

### What to implement

1. Create `models.py`, `base_predictor.py`, `classifier.py`, `explainer.py`,
   `mc_dropout.py` as stubs per the contracts above.
2. Stub `create_app` in `app.py` behind the real `serve` extra guard.
3. Rewire `cli.py` imports off `predictor` and onto the new modules; the three
   subcommands may be stubs (`raise NotImplementedError` inside the command bodies) but
   must keep the `_typer is None` guard and `main()` behavior intact.
4. Rewrite `__init__.py` exports as above.
5. Leave `predictor.py` in place untouched (deleted in #6).

### Tests

Rewrite `test_public_api.py` to assert the **new** surface; drop assertions tied to the
old monolith (`Predictor`, `pull_model`, `_PredictorState`, `predictor` module patching).
The slice issues own the behavioral tests for each class; the public-API test asserts only
import/shape contracts:

- `import radiologist.inference` succeeds.
- `set(pkg.__all__)` equals the post-epic set, and every name is importable.
- The four result dataclasses are constructable with the documented fields.
- `BasePredictor`, `Classifier`, `Explainer`, `MCDropoutPredictor` are importable and the
  subclass relationships hold (`issubclass(Explainer, Classifier)`,
  `issubclass(Classifier, BasePredictor)`, `issubclass(MCDropoutPredictor, BasePredictor)`).
- `create_app()` raises `RuntimeError` naming `serve` when fastapi is absent (patch
  `app._fastapi`).

> Do not assert `NotImplementedError` on the inference verbs here — those stubs are
> replaced behaviorally by #2–#5; pinning them in the public-API test would create churn.

### Acceptance criteria

- [ ] `import radiologist.inference` succeeds and `__all__` equals the post-epic set; `Predictor` and `pull_model` are absent from `__all__`.
- [ ] `Classifier`, `Explainer`, `MCDropoutPredictor` import and their subclass relationships hold.
- [ ] `create_app()` raises `RuntimeError` naming `serve` when fastapi is absent.
- [ ] mypy clean; pytest green (no new behavioral tests; existing public-API test rewritten to the new surface).
