# Inference CLI Reference

The unified `radiologist` CLI (Typer-based, entry point
`radiologist.cli.main:main`, package `radiologist-cli`) exposes the `infer`
command group with four commands: [`predict`](#predict), [`explain`](#explain),
[`uncertainty`](#uncertainty), and [`serve`](#serve) — invoked as
`radiologist infer <command>`. It requires the `inference` extra
(`pip install 'radiologist-cli[inference]'`); `serve` additionally requires
the `serve` extra on `radiologist-inference` (fastapi, uvicorn).

Every command prints a single keyed record to stdout via
`radiologist.utils.cli.emit` — `kv` (default, `key=value` lines), `json`, or
`yaml`, selected with the global `--output`/`-o` flag or the
`RADIOLOGIST_OUTPUT` environment variable (e.g.
`radiologist --output json infer predict ...`).

## Registry selector vs. local path

Every command that loads a model shares the same dispatch rule, implemented
via `verbs.load_predictor`/`verbs.get_verb`:

- **Local path only** — pass `--path` with no selector flag to load an ONNX
  file directly from disk via `BasePredictor.from_path`.
- **Registry selector** — pass any of `--run-id`, `--tags`, `--groups`, or
  `--metric` *together with* `--path` to resolve and download an artifact
  from the W&B Model Registry via `BasePredictor.from_selector`. `--path`
  supplies the entity/project (e.g. `entity/project`) the selector resolves
  against; the resolved file is saved into `--local-dir` (defaults to `.`).
  A registry selector given without `--path` raises an error rather than
  silently falling back to W&B's ambient default entity/project.
- Providing neither `--path` nor any selector flag raises an error (except
  for `serve`, where omitting both starts the server with no model loaded —
  see below).

The resolved artifact's provenance — its fully qualified registry name and
version — is surfaced in every command's output record as
`model_qualified_name` and `model_version`. Both are `null` when the
predictor was loaded via `--path` alone (no registry lookup took place).

`uncertainty` loads a single-session `MCDropoutPredictor`: the same ONNX
model serves both the deterministic and the stochastic MC-Dropout forward
passes, so there is no separate MC-Dropout model/selector to provide.

All four commands also accept `--mean`, `--std`, and `--input-shape`,
forwarded straight to `from_path`/`from_selector`:

- `--mean` and `--std` must be provided together; omitting both keeps the
  default `/255.0`-only preprocessing.
- `--input-shape` takes a comma-separated `N,C,H,W`, e.g. `1,3,224,224`, and
  is only used as a fallback when the ONNX file's embedded metadata has no
  `input_shape` key.

## `predict`

Run deterministic classification on a chest X-ray image and emit a record
carrying the predicted class label and per-class probabilities.

```bash
# Local ONNX file
radiologist infer predict chest_xray.png --path model.onnx

# Registry-backed resolution
radiologist infer predict chest_xray.png --path entity/project \
    --run-id abc123 --local-dir ./models

# With explicit normalization and input shape
radiologist infer predict chest_xray.png --path model.onnx \
    --mean 128 --std 65 --input-shape 1,3,224,224
```

Emitted keys: `predicted_class`, `probabilities` (nested `Dict[str, float]`),
`model_qualified_name`, `model_version` (both `null` unless a registry
selector was used).

| Option | Description |
|---|---|
| `image_path` (argument) | Path to the input chest X-ray image. |
| `--path` | Path to the deterministic ONNX model. |
| `--run-id` | W&B run ID identifying the registry artifact to resolve. Requires `--path` (supplies the entity/project). |
| `--tags` | Registry artifact tag(s) to filter by when `--run-id` is not used. Repeatable. |
| `--groups` | Registry artifact group(s) to filter by when `--run-id` is not used. Repeatable. |
| `--metric` | Metric name used to pick the best-scoring artifact among candidates matching `--tags`/`--groups`. |
| `--local-dir` | Local directory the resolved registry artifact is downloaded into. Ignored when `--path` is used with no registry selector. Defaults to `.`. |
| `--mean` | Normalization mean (requires `--std`). |
| `--std` | Normalization std (requires `--mean`). |
| `--input-shape` | Fallback `[N,C,H,W]` as comma-separated ints, e.g. `1,3,224,224`. |

## `explain`

Produce a Score-CAM saliency map for a chest X-ray image, emitting a record
with the predicted class and either the saved saliency map's path or its
shape.

```bash
# Save the saliency map to disk
radiologist infer explain chest_xray.png --path model.onnx --out saliency.npy

# Emit the saliency map shape only, writing no file
radiologist infer explain chest_xray.png --path model.onnx
```

Emitted keys: `predicted_class`, `saliency_shape` (list of ints),
`saliency_path` (`null` unless `--out` was given), `model_qualified_name`,
`model_version` (both `null` unless a registry selector was used).

| Option | Description |
|---|---|
| `image_path` (argument) | Path to the input chest X-ray image. |
| `--path` | Path to the deterministic ONNX model. |
| `--run-id` | W&B run ID identifying the registry artifact to resolve. Requires `--path` (supplies the entity/project). |
| `--tags` | Registry artifact tag(s) to filter by when `--run-id` is not used. Repeatable. |
| `--groups` | Registry artifact group(s) to filter by when `--run-id` is not used. Repeatable. |
| `--metric` | Metric name used to pick the best-scoring artifact among candidates matching `--tags`/`--groups`. |
| `--local-dir` | Local directory the resolved registry artifact is downloaded into. Ignored when `--path` is used with no registry selector. Defaults to `.`. |
| `--out` | Path to save the saliency map as a `.npy` file. When omitted, only the shape is printed. |
| `--mean` | Normalization mean (requires `--std`). |
| `--std` | Normalization std (requires `--mean`). |
| `--input-shape` | Fallback `[N,C,H,W]` as comma-separated ints, e.g. `1,3,224,224`. |

## `uncertainty`

Estimate MC-Dropout predictive uncertainty for a chest X-ray image: runs
`--n-passes` stochastic forward passes and emits a record with the predicted
class, per-class mean probability and standard deviation, and the overall
predictive entropy.

```bash
# Local ONNX file
radiologist infer uncertainty chest_xray.png --path model.onnx

# Registry-backed resolution
radiologist infer uncertainty chest_xray.png --path entity/project \
    --run-id abc123 --local-dir ./models

# More stochastic passes for a tighter estimate
radiologist infer uncertainty chest_xray.png --path model.onnx --n-passes 100
```

Emitted keys: `predicted_class`, `n_passes`, `predictive_entropy`,
`mean_probabilities`, `std_probabilities` (both nested `Dict[str, float]`),
`model_qualified_name`, `model_version` (both `null` unless a registry
selector was used).

| Option | Description |
|---|---|
| `image_path` (argument) | Path to the input chest X-ray image. |
| `--path` | Path to the MC-Dropout ONNX model. |
| `--run-id` | W&B run ID identifying the registry artifact to resolve. Requires `--path` (supplies the entity/project). |
| `--tags` | Registry artifact tag(s) to filter by when `--run-id` is not used. Repeatable. |
| `--groups` | Registry artifact group(s) to filter by when `--run-id` is not used. Repeatable. |
| `--metric` | Metric name used to pick the best-scoring artifact among candidates matching `--tags`/`--groups`. |
| `--local-dir` | Local directory the resolved registry artifact is downloaded into. Ignored when `--path` is used with no registry selector. Defaults to `.`. |
| `--n-passes` | Number of stochastic forward passes. Defaults to `30`. |
| `--mean` | Normalization mean (requires `--std`). |
| `--std` | Normalization std (requires `--mean`). |
| `--input-shape` | Fallback `[N,C,H,W]` as comma-separated ints, e.g. `1,3,224,224`. |

## `serve`

Launch the FastAPI inference server via uvicorn. Loads a predictor matching
exactly one of `--predict`/`--explain`/`--uncertainty` (default `explain`,
which also satisfies the `Classifier` interface) using the same
registry-selector-vs-local-path dispatch as the other commands. Emits a
record before the server starts accepting connections.

```bash
# Serve a local model (default verb: explain)
radiologist infer serve --path model.onnx --host 0.0.0.0 --port 8000

# Serve a registry-resolved model as a Classifier
radiologist infer serve --path entity/project \
    --run-id abc123 --local-dir ./models --predict

# Start with no model loaded (routes return 503 until one is available)
radiologist infer serve
```

If neither `--path` nor a registry selector is provided, the server starts
with no predictor loaded: `/predict`, `/explain`, and `/uncertainty` each
respond with a `503` "no model loaded" error until a predictor becomes
available. `/healthz` (liveness) and `/readyz` (readiness) are always
available regardless of predictor state.

Emitted keys: `host`, `port`, `verb`, `model_path` (`null` unless `--path`
was given), `model_run_id` (`null` unless `--run-id` was given),
`model_qualified_name`, `model_version` (both `null` unless a registry
selector was used).

| Option | Description |
|---|---|
| `--path` | Path to the deterministic ONNX model. |
| `--run-id` | W&B run ID identifying the registry artifact to resolve. Requires `--path` (supplies the entity/project). |
| `--tags` | Registry artifact tag(s) to filter by when `--run-id` is not used. Repeatable. |
| `--groups` | Registry artifact group(s) to filter by when `--run-id` is not used. Repeatable. |
| `--metric` | Metric name used to pick the best-scoring artifact among candidates matching `--tags`/`--groups`. |
| `--local-dir` | Local directory the resolved registry artifact is downloaded into. Ignored when `--path` is used with no registry selector. Defaults to `.`. |
| `--host` | Host interface to bind the HTTP server to. Defaults to `127.0.0.1`. |
| `--port` | TCP port to bind the HTTP server to. Defaults to `8000`. |
| `--predict` | Serve a `Classifier` (predict verb). |
| `--explain` | Serve an `Explainer` (explain verb, default). |
| `--uncertainty` | Serve an `MCDropoutPredictor` (uncertainty verb). |

## Python API equivalent

Every command is a thin wrapper over `radiologist-inference`'s predictor
classes, so any workflow above can also be scripted directly:

```python
from radiologist.inference import Classifier, Explainer, MCDropoutPredictor

predictor = Classifier.from_path("model.onnx")
result = predictor.predict(image)

explainer = Explainer.from_selector(path="entity/project", tags=["prod"])
explanation = explainer.explain(image)
```

`create_app(...)` builds the same FastAPI app `serve` runs under uvicorn. See
[radiologist-inference/README.md § Using the public API](../pkg/inference.md#using-the-public-api)
for the full constructor/method reference.
