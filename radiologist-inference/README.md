# radiologist-inference

ONNX inference and serving for the radiologist pipeline. Pulls trained models from the W&B Model Registry, runs deterministic classification, Score-CAM saliency, and MC-Dropout uncertainty estimation via ONNX Runtime, and optionally exposes a FastAPI HTTP server and a Typer CLI.

## Installation

### Hard dependencies (always installed)

```bash
pip install radiologist-inference
```

Installs: `numpy`, `Pillow`, `onnxruntime`.

### Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `registry` | `wandb` | `BasePredictor.from_registry` |
| `serve` | `fastapi`, `uvicorn`, `python-multipart` | `create_app`, HTTP server |
| `cli` | `typer` | `radiologist` CLI entry point |
| `all` | all of the above | everything |

```bash
pip install "radiologist-inference[all]"
```

## Quick start

### Library

The predictor hierarchy is capability-based: `Classifier` adds deterministic
prediction, `Explainer` (a `Classifier`) adds Score-CAM explanation, and
`MCDropoutPredictor` adds MC-Dropout uncertainty estimation. Pick the class
matching the capabilities you need.

```python
from radiologist.inference import Classifier, Explainer, MCDropoutPredictor

# Deterministic prediction only
classifier = Classifier.from_path(det_path="model.onnx")
result = classifier.predict("chest_xray.png")
print(result.predicted_class)          # "NORMAL"
print(result.probabilities)            # {"NORMAL": 0.93, "PNEUMONIA": 0.07}

# Score-CAM explanation (Explainer inherits predict from Classifier)
explainer = Explainer.from_path(det_path="model.onnx")
explanation = explainer.explain("chest_xray.png")
print(explanation.saliency_map.shape)  # (H, W)

# MC-Dropout uncertainty (requires mcd_path)
mcd_predictor = MCDropoutPredictor.from_path(
    det_path="model.onnx", mcd_path="model_mcd.onnx"
)
uncertainty = mcd_predictor.predict_with_uncertainty("chest_xray.png", n_passes=30)
print(uncertainty.predictive_entropy)
print(uncertainty.std_per_class)
```

Download a model from the W&B Model Registry via `from_registry` (requires
the `registry` extra):

```python
classifier = Classifier.from_registry(
    artifact_path="entity/project/model-name:v1",
    local_dir="./models",
)
```

### HTTP server (requires `serve` extra)

`create_app` dispatches routes based on `isinstance` checks against the
injected predictor: any `Classifier` gets `/predict`, an `Explainer` also
gets `/explain`, and an `MCDropoutPredictor` gets `/uncertainty`. Passing no
predictor wires every route, each guarded by a 503 until one is injected.

```python
from radiologist.inference import Explainer, create_app

predictor = Explainer.from_path("model.onnx")
app = create_app(predictor=predictor)
# Pass app to uvicorn or any ASGI server
```

```bash
uvicorn mymodule:app --host 0.0.0.0 --port 8000
```

Routes:

| Method | Path | Description |
|---|---|---|
| `GET` | `/healthz` | Liveness check |
| `POST` | `/predict` | Classify a chest X-ray image (multipart upload) |
| `POST` | `/explain` | Return Score-CAM saliency map |
| `POST` | `/uncertainty` | MC-Dropout uncertainty estimation |

### CLI (requires `cli` extra)

```bash
# Classify a chest X-ray
radiologist predict chest_xray.png --model model.onnx

# Score-CAM explanation
radiologist explain chest_xray.png --model model.onnx --out saliency.npy

# MC-Dropout uncertainty
radiologist uncertainty chest_xray.png --model model.onnx --mcd-model model_mcd.onnx
```

## Public API reference

### `BasePredictor`

Common loading surface shared by every predictor class.

| Method | Signature | Description |
|---|---|---|
| `from_path` | `(det_path: str, mcd_path: Optional[str] = None) -> BasePredictor` | Load from local ONNX files |
| `from_registry` | `(artifact_path: str, local_dir: str, registry=None) -> BasePredictor` | Download from W&B Registry and load; requires `registry` extra |

### `Classifier(BasePredictor)`

| Method | Signature | Description |
|---|---|---|
| `predict` | `(image, deployment_prior=None) -> Prediction` | Deterministic inference; `image` accepts a file path, NumPy HWC uint8 array, or PIL Image |

### `Explainer(Classifier)`

| Method | Signature | Description |
|---|---|---|
| `explain` | `(image) -> Explanation` | Score-CAM saliency map for the given image; `predict` is inherited from `Classifier` |

### `MCDropoutPredictor(BasePredictor)`

| Method | Signature | Description |
|---|---|---|
| `predict_with_uncertainty` | `(image, n_passes: int = 30) -> UncertaintyResult` | MC-Dropout stochastic inference; requires `mcd_path` at load time |

### `score_cam`

```python
score_cam(feature_maps: np.ndarray, logits: np.ndarray) -> np.ndarray
```

Compute a Score-CAM saliency map from feature maps `(C, H, W)` and logits `(num_classes,)`. Returns a `(H, W)` array with values in `[0, 1]`.

### `mc_dropout_predict`

```python
mc_dropout_predict(
    session: ort.InferenceSession,
    image: np.ndarray,
    n_passes: int = 30,
) -> UncertaintyResult
```

Run `n_passes` stochastic forward passes through an MC-Dropout ONNX model and aggregate uncertainty statistics.

### `create_app`

```python
create_app(predictor: Optional[BasePredictor] = None) -> FastAPI
```

Create and return the FastAPI application, wiring routes to the injected
predictor's capabilities (see [HTTP server](#http-server-requires-serve-extra)
above). Requires the `serve` extra.

### Result dataclasses

| Class | Fields |
|---|---|
| `Prediction` | `probabilities: Dict[str, float]`, `predicted_class: str` |
| `Explanation` | `saliency_map: np.ndarray`, `predicted_class: str` |
| `UncertaintyResult` | `mean_probabilities: Dict[str, float]`, `std_per_class: Dict[str, float]`, `predictive_entropy: float`, `n_passes: int` |
| `ModelMetadata` | `classes: List[str]`, `input_shape: List[int]`, `cam_target_layer: str`, `output_names: List[str]`, `mc_dropout: bool` |

## Development setup

```bash
pyenv activate radiologist
uv sync --active --extra all --all-groups
uv run --active pytest radiologist-inference/tests -q
```
