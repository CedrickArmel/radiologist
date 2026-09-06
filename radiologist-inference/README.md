# radiologist-inference

[![ci](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml/badge.svg)](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CedrickArmel/radiologist/branch/main/graph/badge.svg)](https://codecov.io/gh/CedrickArmel/radiologist)
[![PyPI](https://img.shields.io/pypi/v/radiologist-inference)](https://pypi.org/project/radiologist-inference/)
![tested on](https://img.shields.io/badge/tested%20on-ubuntu--latest%20%7C%20python%203.10-blue)

ONNX inference and serving for the radiologist pipeline. Pulls trained models from the W&B Model Registry, runs deterministic classification, Score-CAM saliency, and MC-Dropout uncertainty estimation via ONNX Runtime, and optionally exposes a FastAPI HTTP server. Its CLI is exposed through the `infer` command group of the unified `radiologist` CLI, see [radiologist-cli](../radiologist-cli/README.md).

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
| `serve` | `fastapi`, `uvicorn`, `python-multipart`, `prometheus-client` | `create_app`, HTTP server, `GET /metrics` |
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
| `GET` | `/readyz` | Readiness check; 503 until a predictor is loaded |
| `POST` | `/predict` | Classify a chest X-ray image (multipart upload) |
| `POST` | `/explain` | Return Score-CAM saliency map |
| `POST` | `/uncertainty` | MC-Dropout uncertainty estimation |
| `GET` | `/metrics` | Prometheus exposition of the metric catalogue below |

### Metrics (`GET /metrics`, requires `serve` extra)

Instrumentation is always-on whenever `prometheus-client` is importable — no
CLI flag, no configuration. `GET /metrics` returns a
`text/plain; version=0.0.4; charset=utf-8` Prometheus exposition payload of
this application's own `CollectorRegistry`. When `prometheus-client` is not
installed, `GET /metrics` is not wired at all and the route 404s cleanly;
every `Metrics` recording method silently no-ops instead of raising, so
request handling is unaffected either way.

The registry is **per-application and per-process**: each call to
`create_app` builds a fresh `CollectorRegistry`, so only these ten families
are exposed — there are no `process_*` / `python_gc_*` collectors. The
deployment model is assumed **single-worker**;
`prometheus_client.multiprocess` (multi-process/multi-worker aggregation) is
not supported. Scrape traffic against `/metrics` itself is excluded from all
counters, histograms and gauges below — it is not counted, timed, or tracked
in flight.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `inference_requests_total` | Counter | `route`, `status` | Total inference API requests |
| `inference_request_duration_seconds` | Histogram | `route` | Wall-clock request duration, in seconds |
| `inference_requests_in_progress` | Gauge | `route` | Requests currently being served |
| `inference_errors_total` | Counter | `route`, `error_type` | Request-level errors |
| `inference_input_image_size_bytes` | Histogram | — | Uploaded input image size, in bytes |
| `inference_input_image_width_pixels` | Histogram | — | Pre-resize input image width, in pixels |
| `inference_input_image_height_pixels` | Histogram | — | Pre-resize input image height, in pixels |
| `inference_predicted_class_total` | Counter | `class` | Predictions per predicted class |
| `inference_confidence` | Histogram | — | Maximum predicted class probability |
| `inference_predictive_entropy` | Histogram | — | Predictive entropy of the mean MC-Dropout prediction |
| `inference_uncertainty_std_max` | Histogram | — | Maximum per-class std across MC-Dropout passes |

`route` and `error_type` are closed, bounded-cardinality label sets — never
caller-controlled text:

- `route` is one of `/predict`, `/explain`, `/uncertainty`, `/healthz`,
  `/readyz`, `/metrics`, or the fallback value `unmatched` for any other
  path.
- `error_type` is one of `invalid_image`, `empty_file`, `no_model_loaded`,
  `validation_error`.

Two scope reconciliations, recorded here so they are not re-litigated:

- `/explain` does not observe `inference_confidence` — `Explanation` carries
  no probability vector to derive a confidence value from.
- `/uncertainty` does not increment `inference_predicted_class_total` —
  `UncertaintyResult` has no `predicted_class` field.

### CLI

This package no longer ships its own console script. Its commands are the
`infer` group of the unified `radiologist` CLI — install
`radiologist-cli[inference]` and see
[docs/reference/cli-inference.md](../docs/reference/cli-inference.md) for
the full command reference and examples (`radiologist infer predict ...`,
`radiologist infer explain ...`, `radiologist infer uncertainty ...`,
`radiologist infer serve ...`).

## Using the public API

### `BasePredictor`

Common loading surface shared by every predictor class.

| Method | Signature | Description |
|---|---|---|
| `from_path` | `(model_path: str, mean: Optional[float] = None, std: Optional[float] = None, input_shape: Optional[List[int]] = None) -> BasePredictor` | Load from a local ONNX file |
| `from_registry` | `(artifact_path: str, local_dir: str, registry=None, mean: Optional[float] = None, std: Optional[float] = None, input_shape: Optional[List[int]] = None) -> BasePredictor` | Download from W&B Registry and load; requires `registry` extra |
| `from_selector` | `(selector, local_dir: str, registry=None, mean: Optional[float] = None, std: Optional[float] = None, input_shape: Optional[List[int]] = None) -> BasePredictor` | Resolve a `RegistrySelector` against a registry, download, and load; requires `registry` extra — the path the CLI takes |
| `provenance` | `-> Optional[ArtifactRef]` (property) | The registry artifact this predictor was resolved from; `None` for a predictor built with `from_path` |

### `Classifier(BasePredictor)`

| Method | Signature | Description |
|---|---|---|
| `predict` | `(image, deployment_prior=None) -> Prediction` | Deterministic inference; `image` accepts a file path, NumPy HWC uint8 array, or PIL Image |

### `Explainer(Classifier)`

| Method | Signature | Description |
|---|---|---|
| `explain` | `(image, deployment_prior=None) -> Explanation` | Score-CAM saliency map for the given image; `predict` is inherited from `Classifier` |

### `MCDropoutPredictor(BasePredictor)`

| Method | Signature | Description |
|---|---|---|
| `predict_with_uncertainty` | `(image, n_passes: int = 30) -> UncertaintyResult` | MC-Dropout stochastic inference |

### `score_cam`

```python
score_cam(feature_maps: np.ndarray, logits: np.ndarray) -> np.ndarray
```

Compute a Score-CAM saliency map from feature maps `(C, H, W)` and logits `(num_classes,)`. `logits` is accepted for interface symmetry and is not read. Returns a `(H, W)` array with values in `[0, 1]`.

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

## Extending this package

**There are no user-swappable Hydra config groups here.** Unlike
`radiologist-core` and `radiologist-etl`, this package ships no config tree and
does not depend on Hydra at all — the `radiologist infer` commands are plain
Typer flags, not `key=value` config overrides. There is nothing to point a
`hydra.searchpath` at.

Two real extension seams exist instead.

### Swapping the model registry backend

Every registry-backed loader takes an optional `registry` argument and defaults
to `WandbRegistry()`. Pass anything satisfying the
[`ModelRegistry`](../radiologist-registry/README.md#extending-the-registry-backend)
protocol — it is a structural `typing.Protocol`, so no inheritance or
registration is needed:

```python
from radiologist.inference import Classifier

predictor = Classifier.from_registry(
    artifact_path="entity/project/model-abc123:best",
    local_dir="/tmp/models",
    registry=MyRegistry(),   # any object matching radiologist.registry.ModelRegistry
)
```

This is the seam to use for an on-prem artifact store, a test double, or a
caching layer in front of W&B.

### Adding a verb by subclassing

`BasePredictor` owns loading, provenance, and preprocessing; each subclass adds
one verb (`Classifier.predict`, `Explainer.explain`,
`MCDropoutPredictor.predict_with_uncertainty`). Construction goes through
`cls.__new__(cls)` inside the three classmethods, so a subclass inherits
`from_path` / `from_registry` / `from_selector` for free:

```python
from radiologist.inference import Classifier


class TopKClassifier(Classifier):
    def predict_top_k(self, image, k: int = 3):
        prediction = self.predict(image)
        ranked = sorted(prediction.probabilities.items(), key=lambda kv: -kv[1])
        return ranked[:k]


predictor = TopKClassifier.from_path("/tmp/models/model.onnx")
```

The HTTP layer is not currently extensible in the same way: `create_app`
dispatches routes by `isinstance` over the three shipped predictor classes, so a
subclass gets its parent's routes, not a new one. The internal verb table in
`radiologist.inference.verbs` is module-private with no registration function —
it is not a supported extension point today.

## Development setup

```bash
pyenv activate radiologist
uv sync --active --extra all --all-groups
uv run --active pytest radiologist-inference/radiologist_inference_tests -q
```
