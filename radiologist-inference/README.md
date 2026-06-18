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
| `registry` | `wandb` | `pull_model`, `Predictor.from_registry` |
| `serve` | `fastapi`, `uvicorn`, `python-multipart` | `create_app`, HTTP server |
| `cli` | `typer` | `radiologist` CLI entry point |
| `all` | all of the above | everything |

```bash
pip install "radiologist-inference[all]"
```

## Quick start

### Library

```python
from radiologist.inference import Predictor

# Load from local ONNX files
predictor = Predictor.from_path(
    det_path="model.onnx",
    mcd_path="model_mcd.onnx",  # optional: enables predict_with_uncertainty
)

# Deterministic prediction
result = predictor.predict("chest_xray.png")
print(result.predicted_class)          # "NORMAL"
print(result.probabilities)            # {"NORMAL": 0.93, "PNEUMONIA": 0.07}

# Score-CAM saliency map
explanation = predictor.explain("chest_xray.png")
print(explanation.saliency_map.shape)  # (H, W)

# MC-Dropout uncertainty (requires mcd_path)
uncertainty = predictor.predict_with_uncertainty("chest_xray.png", n_passes=30)
print(uncertainty.predictive_entropy)
print(uncertainty.std_per_class)
```

Download a model from the W&B Model Registry first:

```python
from radiologist.inference import pull_model

local_path = pull_model(
    artifact_path="entity/project/model-name:v1",
    local_dir="./models",
)
predictor = Predictor.from_path(det_path=local_path)
# or equivalently:
predictor = Predictor.from_registry(
    artifact_path="entity/project/model-name:v1",
    local_dir="./models",
)
```

### HTTP server (requires `serve` extra)

```python
from radiologist.inference import create_app, Predictor

predictor = Predictor.from_path("model.onnx")
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

# With MC-Dropout model
radiologist predict chest_xray.png --model model.onnx --mcd-model model_mcd.onnx

# Download model from W&B Registry
radiologist pull entity/project/model-name:v1 --local-dir ./models
```

## Public API reference

### `Predictor`

Facade for ONNX-backed classification.

| Method | Signature | Description |
|---|---|---|
| `from_path` | `(det_path: str, mcd_path: Optional[str] = None) -> Predictor` | Load from local ONNX files |
| `from_registry` | `(artifact_path: str, local_dir: str) -> Predictor` | Download from W&B Registry and load; requires `registry` extra |
| `predict` | `(image, deployment_prior=None) -> Prediction` | Deterministic inference; `image` accepts a file path, NumPy HWC uint8 array, or PIL Image |
| `explain` | `(image) -> Explanation` | Score-CAM saliency map for the given image |
| `predict_with_uncertainty` | `(image, n_passes: int = 30) -> UncertaintyResult` | MC-Dropout stochastic inference; requires `mcd_path` at load time |

### `pull_model`

```python
pull_model(artifact_path: str, local_dir: str) -> str
```

Download an ONNX model artifact from the W&B Model Registry. Returns the local path to the `.onnx` file. Requires the `registry` extra.

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
create_app(predictor: Optional[Predictor] = None) -> FastAPI
```

Create and return the FastAPI application. Requires the `serve` extra.

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
