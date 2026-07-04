# Inference CLI Reference

The `radiologist` CLI (Typer-based, entry point `radiologist.inference.cli:main`)
exposes four commands: [`predict`](#predict), [`explain`](#explain),
[`uncertainty`](#uncertainty), and [`serve`](#serve). It requires the `cli`
extra (`pip install 'radiologist-inference[cli]'`); `serve` additionally
requires the `serve` extra (fastapi, uvicorn).

## Registry selector vs. local path

Every command that loads a model shares the same dispatch rule, implemented
via `verbs.load_predictor`/`verbs.get_verb`:

- **Local path** — pass `--model` to load an ONNX file directly from disk via
  `BasePredictor.from_path`.
- **Registry selector** — pass any of `--run-id`, `--tags`, `--groups`, or
  `--metric` (without `--model`) to resolve and download an artifact from the
  W&B Model Registry via `BasePredictor.from_selector`. The resolved file is
  saved into `--local-dir` (defaults to `.`).
- Exactly one of the two strategies must be usable. Providing neither
  `--model` nor any selector flag raises an error (except for `serve`, where
  omitting both starts the server with no model loaded — see below).

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

Run deterministic classification on a chest X-ray image and print the
predicted class label with per-class probabilities.

```bash
# Local ONNX file
radiologist predict chest_xray.png --model model.onnx

# Registry-backed resolution
radiologist predict chest_xray.png --run-id abc123 --local-dir ./models

# With explicit normalization and input shape
radiologist predict chest_xray.png --model model.onnx \
    --mean 128 --std 65 --input-shape 1,3,224,224
```

| Option | Description |
|---|---|
| `image_path` (argument) | Path to the input chest X-ray image. |
| `--model` | Path to the deterministic ONNX model. |
| `--run-id` | W&B run ID identifying the registry artifact to resolve. Mutually exclusive with `--model`. |
| `--tags` | Registry artifact tag(s) to filter by when `--run-id` is not used. Repeatable. |
| `--groups` | Registry artifact group(s) to filter by when `--run-id` is not used. Repeatable. |
| `--metric` | Metric name used to pick the best-scoring artifact among candidates matching `--tags`/`--groups`. |
| `--local-dir` | Local directory the resolved registry artifact is downloaded into. Ignored when `--model` is used. Defaults to `.`. |
| `--mean` | Normalization mean (requires `--std`). |
| `--std` | Normalization std (requires `--mean`). |
| `--input-shape` | Fallback `[N,C,H,W]` as comma-separated ints, e.g. `1,3,224,224`. |

## `explain`

Produce a Score-CAM saliency map for a chest X-ray image, printing the
predicted class and either saving the saliency map or printing its shape.

```bash
# Save the saliency map to disk
radiologist explain chest_xray.png --model model.onnx --out saliency.npy

# Print the saliency map shape only
radiologist explain chest_xray.png --model model.onnx
```

| Option | Description |
|---|---|
| `image_path` (argument) | Path to the input chest X-ray image. |
| `--model` | Path to the deterministic ONNX model. |
| `--run-id` | W&B run ID identifying the registry artifact to resolve. Mutually exclusive with `--model`. |
| `--tags` | Registry artifact tag(s) to filter by when `--run-id` is not used. Repeatable. |
| `--groups` | Registry artifact group(s) to filter by when `--run-id` is not used. Repeatable. |
| `--metric` | Metric name used to pick the best-scoring artifact among candidates matching `--tags`/`--groups`. |
| `--local-dir` | Local directory the resolved registry artifact is downloaded into. Ignored when `--model` is used. Defaults to `.`. |
| `--out` | Path to save the saliency map as a `.npy` file. When omitted, only the shape is printed. |
| `--mean` | Normalization mean (requires `--std`). |
| `--std` | Normalization std (requires `--mean`). |
| `--input-shape` | Fallback `[N,C,H,W]` as comma-separated ints, e.g. `1,3,224,224`. |

## `uncertainty`

Estimate MC-Dropout predictive uncertainty for a chest X-ray image: runs
`--n-passes` stochastic forward passes and prints per-class mean probability
with standard deviation, plus the overall predictive entropy.

```bash
# Local ONNX file
radiologist uncertainty chest_xray.png --model model.onnx

# Registry-backed resolution
radiologist uncertainty chest_xray.png --run-id abc123 --local-dir ./models

# More stochastic passes for a tighter estimate
radiologist uncertainty chest_xray.png --model model.onnx --n-passes 100
```

| Option | Description |
|---|---|
| `image_path` (argument) | Path to the input chest X-ray image. |
| `--model` | Path to the MC-Dropout ONNX model. |
| `--run-id` | W&B run ID identifying the registry artifact to resolve. Mutually exclusive with `--model`. |
| `--tags` | Registry artifact tag(s) to filter by when `--run-id` is not used. Repeatable. |
| `--groups` | Registry artifact group(s) to filter by when `--run-id` is not used. Repeatable. |
| `--metric` | Metric name used to pick the best-scoring artifact among candidates matching `--tags`/`--groups`. |
| `--local-dir` | Local directory the resolved registry artifact is downloaded into. Ignored when `--model` is used. Defaults to `.`. |
| `--n-passes` | Number of stochastic forward passes. Defaults to `30`. |
| `--mean` | Normalization mean (requires `--std`). |
| `--std` | Normalization std (requires `--mean`). |
| `--input-shape` | Fallback `[N,C,H,W]` as comma-separated ints, e.g. `1,3,224,224`. |

## `serve`

Launch the FastAPI inference server via uvicorn. Loads a predictor matching
exactly one of `--predict`/`--explain`/`--uncertainty` (default `explain`,
which also satisfies the `Classifier` interface) using the same
registry-selector-vs-local-path dispatch as the other commands.

```bash
# Serve a local model (default verb: explain)
radiologist serve --model model.onnx --host 0.0.0.0 --port 8000

# Serve a registry-resolved model as a Classifier
radiologist serve --run-id abc123 --local-dir ./models --predict

# Start with no model loaded (routes return 503 until one is available)
radiologist serve
```

If neither `--model` nor a registry selector is provided, the server starts
with no predictor loaded: `/predict`, `/explain`, and `/uncertainty` each
respond with a `503` "no model loaded" error until a predictor becomes
available. `/healthz` (liveness) and `/readyz` (readiness) are always
available regardless of predictor state.

| Option | Description |
|---|---|
| `--model` | Path to the deterministic ONNX model. |
| `--run-id` | W&B run ID identifying the registry artifact to resolve. Mutually exclusive with `--model`. |
| `--tags` | Registry artifact tag(s) to filter by when `--run-id` is not used. Repeatable. |
| `--groups` | Registry artifact group(s) to filter by when `--run-id` is not used. Repeatable. |
| `--metric` | Metric name used to pick the best-scoring artifact among candidates matching `--tags`/`--groups`. |
| `--local-dir` | Local directory the resolved registry artifact is downloaded into. Ignored when `--model` is used. Defaults to `.`. |
| `--host` | Host interface to bind the HTTP server to. Defaults to `127.0.0.1`. |
| `--port` | TCP port to bind the HTTP server to. Defaults to `8000`. |
| `--predict` | Serve a `Classifier` (predict verb). |
| `--explain` | Serve an `Explainer` (explain verb, default). |
| `--uncertainty` | Serve an `MCDropoutPredictor` (uncertainty verb). |
