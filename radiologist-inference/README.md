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
classifier = Classifier.from_path(model_path="model.onnx")
result = classifier.predict("chest_xray.png")
print(result.predicted_class)          # "NORMAL"
print(result.probabilities)            # {"NORMAL": 0.93, "PNEUMONIA": 0.07}

# Score-CAM explanation (Explainer inherits predict from Classifier)
explainer = Explainer.from_path(model_path="model.onnx")
explanation = explainer.explain("chest_xray.png")
print(explanation.saliency_map.shape)  # (H, W)

# MC-Dropout uncertainty (loaded from a stochastic MC-Dropout ONNX model)
mcd_predictor = MCDropoutPredictor.from_path(model_path="model_mcd.onnx")
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

`from_path`/`from_registry` accept optional `mean`, `std`, and `input_shape`
kwargs. This is a **behavior-relevant** addition: existing callers that omit
them keep getting exactly today's `/255.0`-only preprocessing (fully
backward compatible). Passing both `mean` and `std` applies
`(arr / 255 - mean) / std` instead, letting inference match the model's
actual training-time normalization without a custom fork — e.g. this
project's `radiologist-core` training pipeline uses `Normalize(mean=[128],
std=[65])` after `[0, 1]`-scaling, so `mean=128.0, std=65.0` reproduces
train/serve-consistent preprocessing:

```python
classifier = Classifier.from_path(
    model_path="model.onnx", mean=128.0, std=65.0,
)
```

`mean` and `std` must be provided together — passing only one raises
`ValueError` eagerly at load time (`from_path`/`from_registry`/
`from_selector`), before any inference request and, for the registry-backed
loaders, before the ONNX artifact is pulled. `input_shape` (`[N, C, H, W]`)
is a fallback used only when the
ONNX file's embedded metadata has no `input_shape` key; if metadata has no
`input_shape` and none is passed, loading raises `ValueError`.

```python
classifier = Classifier.from_path(
    model_path="model_without_shape_metadata.onnx", input_shape=[1, 3, 224, 224],
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
radiologist uncertainty chest_xray.png --model model_mcd.onnx
```

`predict`, `explain`, and `uncertainty` all accept optional `--mean`,
`--std`, and `--input-shape` flags, threaded straight to `from_path`.
`--input-shape` takes a comma-separated `N,C,H,W`, e.g. `1,3,224,224`.
Omitting all three keeps today's default `/255.0`-only preprocessing:

```bash
radiologist predict chest_xray.png --model model.onnx \
    --mean 128 --std 65 --input-shape 1,3,224,224
```

```bash
# Serve — picks the verb to serve via --predict/--explain/--uncertainty
# (default: explain, preserving today's behavior)
radiologist serve --model model.onnx
radiologist serve --uncertainty --model model_mcd.onnx
radiologist serve --predict --run-id abc123
```

## Public API reference

### `BasePredictor`

Common loading surface shared by every predictor class.

| Method | Signature | Description |
|---|---|---|
| `from_path` | `(model_path: str, mean: Optional[float] = None, std: Optional[float] = None, input_shape: Optional[List[int]] = None) -> BasePredictor` | Load from a local ONNX file |
| `from_registry` | `(artifact_path: str, local_dir: str, registry=None, mean: Optional[float] = None, std: Optional[float] = None, input_shape: Optional[List[int]] = None) -> BasePredictor` | Download from W&B Registry and load; requires `registry` extra |

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
| `predict_with_uncertainty` | `(image, n_passes: int = 30) -> UncertaintyResult` | MC-Dropout stochastic inference |

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
| `ModelMetadata` | `classes: List[str]`, `input_shape: List[int]`, `cam_target_layer: str`, `output_names: List[str]` |

## Development setup

```bash
pyenv activate radiologist
uv sync --active --extra all --all-groups
uv run --active pytest radiologist-inference/tests -q
```
