## 🦴 Predictor decomposition (minimal-impact) — skeleton

### Context

This issue stubs the entire epic surface — the new class hierarchy, the capability-aware app factory, and the three-command CLI — so every slice (#2–#5) can start in parallel blocked only on this issue. No behavior is implemented: every new public method body is `raise NotImplementedError`, every signature is fully type-hinted, and the existing test suite stays green except for the public-API list test, which this issue updates to the new `__all__`. The skeleton is a type-checked contract, not a working system. Minimal-impact: no new source module is created — the whole hierarchy lives in `predictor.py`. See the epic spec (`predictor-minimal-spec.md`).

> Requires: registry epic #85–90 (provides `WandbRegistry`).
> Blocks: #2, #3, #4, #5.

### Module layout

```
radiologist-inference/src/radiologist/inference/
├── predictor.py    # dataclasses, private helpers, BasePredictor + Classifier + Explainer + MCDropoutPredictor, create_app
├── app.py          # _build_app — capability-aware route wiring
├── cam.py          # unchanged: score_cam, score_cam_with_session
├── cli.py          # predict / explain / uncertainty subcommands
├── optional.py     # unchanged: _wandb, _fastapi, _typer sentinels
└── __init__.py     # new __all__
```

### Interface contracts

<!-- All public signatures for the epic. Bodies are `raise NotImplementedError`.
     Full type hints. No logic. Dataclasses keep their existing fields. -->

##### `radiologist-inference/src/radiologist/inference/predictor.py`

```python
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage


@dataclass
class Prediction:
    probabilities: Dict[str, float]
    predicted_class: str


@dataclass
class Explanation:
    saliency_map: np.ndarray
    predicted_class: str


@dataclass
class UncertaintyResult:
    mean_probabilities: Dict[str, float]
    std_per_class: Dict[str, float]
    predictive_entropy: float
    n_passes: int


@dataclass
class ModelMetadata:
    classes: List[str]
    input_shape: List[int]
    cam_target_layer: str
    output_names: List[str]
    mc_dropout: bool


@dataclass
class _PredictorState:
    # contract: one loaded ONNX session + its parsed metadata. Each subclass
    # loads exactly one model into this state.
    session: ort.InferenceSession
    metadata: Dict[str, str]


# --- private helpers: unchanged bodies, reused by all subclasses ---
def _read_metadata(session: ort.InferenceSession) -> Dict[str, str]:
    # contract: returns custom_metadata_map as a plain dict.
    ...

def _preprocess_image(
    image: Union[str, np.ndarray, PILImage.Image],
    input_shape: List[int],
) -> np.ndarray:
    # contract: returns float32 (1, C, H, W) array in [0, 1].
    ...

def _apply_prior_correction(
    softmax: np.ndarray, classes: List[str], prior: Dict[str, float]
) -> np.ndarray:
    # contract: returns renormalized float32 array weighted by prior.
    ...


class BasePredictor:
    """Shared ONNX loading, preprocessing, softmax. Not directly servable."""

    _state: _PredictorState

    @classmethod
    def from_path(cls, model_path: str) -> "BasePredictor":
        # contract: loads ONE ONNX model from a local path, parses metadata, and
        # returns an instance of the concrete cls; raises FileNotFoundError when
        # model_path is absent.
        raise NotImplementedError

    @classmethod
    def from_registry(cls, artifact_path: str, local_dir: str) -> "BasePredictor":
        # contract: resolves a local ONNX path via WandbRegistry.pull(artifact_path,
        # local_dir), then from_path() on it; raises RuntimeError naming the
        # 'registry' extra when wandb is absent.
        raise NotImplementedError


class Classifier(BasePredictor):
    """Deterministic classification from a single deterministic ONNX model."""

    def predict(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
        deployment_prior: Optional[Dict[str, float]] = None,
    ) -> Prediction:
        # contract: per-class probabilities + predicted label; applies
        # deployment_prior when given else embedded training_prior when present.
        raise NotImplementedError


class Explainer(Classifier):
    """Classification + Score-CAM saliency from the same deterministic model."""

    def explain(
        self, image: Union[str, np.ndarray, PILImage.Image]
    ) -> Explanation:
        # contract: (H, W) saliency map in [0, 1] plus predicted label.
        raise NotImplementedError


class MCDropoutPredictor(BasePredictor):
    """MC-Dropout uncertainty from a single stochastic ONNX model."""

    def predict_with_uncertainty(
        self,
        image: Union[str, np.ndarray, PILImage.Image],
        n_passes: int = 30,
    ) -> UncertaintyResult:
        # contract: mean probs, per-class std, predictive entropy, n_passes.
        raise NotImplementedError


# --- stateless helpers: unchanged bodies, stay public ---
def score_cam(feature_maps: np.ndarray, logits: np.ndarray) -> np.ndarray:
    # contract: (C,H,W)+(num_classes,) -> (H,W) saliency in [0,1].
    ...

def mc_dropout_predict(
    session: Any, image: np.ndarray, n_passes: int = 30
) -> UncertaintyResult:
    # contract: runs n stochastic passes and aggregates uncertainty.
    ...


def create_app(predictor: Optional["BasePredictor"] = None) -> Any:
    # contract: smart factory — inspects predictor's concrete type and wires only
    # matching routes; raises RuntimeError naming the 'serve' extra when fastapi
    # is absent. Returns a FastAPI app.
    raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/app.py`

```python
from typing import Any, Optional


def _build_app(fastapi_mod: Any, predictor: Optional[Any]) -> Any:
    # contract: always registers GET /healthz; registers POST /predict when the
    # predictor can classify, POST /explain when it can explain, POST
    # /uncertainty when it can estimate uncertainty. Capability detected by
    # isinstance against Classifier / Explainer / MCDropoutPredictor. A None
    # predictor yields an app whose inference routes return 503.
    raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/cli.py`

```python
# typer app with three commands; `pull` removed.
#   predict     <image_path> --model <det_path>
#   explain     <image_path> --model <det_path>
#   uncertainty <image_path> --model <mcd_path> [--n-passes N]

def main() -> None:
    # contract: raises RuntimeError naming the 'cli' extra when typer is absent.
    raise NotImplementedError
```

##### `radiologist-inference/src/radiologist/inference/__init__.py`

```python
__all__ = [
    "BasePredictor",
    "Classifier",
    "Explainer",
    "MCDropoutPredictor",
    "score_cam",
    "mc_dropout_predict",
    "Prediction",
    "Explanation",
    "UncertaintyResult",
    "ModelMetadata",
    "create_app",
]
```

### Acceptance criteria

- [ ] `BasePredictor`, `Classifier`, `Explainer`, `MCDropoutPredictor`, `create_app`, `score_cam`, `mc_dropout_predict`, and the four result dataclasses are all importable from `radiologist.inference`.
- [ ] `Predictor` and `pull_model` are no longer importable from `radiologist.inference`.
- [ ] `set(radiologist.inference.__all__)` equals the new list above.
- [ ] mypy clean; pytest green (stubs typecheck; the public-API test is updated to the new `__all__`; no new behavioral tests).
