# Registry CLI Reference

`radiologist-registry` is a Typer-based CLI wrapping `WandbRegistry`. Every
command constructs a `WandbRegistry()` internally — no separate wiring is
required. Errors surface as `Error: {message}` on stderr with a non-zero
exit code.

Install the CLI and W&B extras first:

```bash
pip install "radiologist-registry[cli,wandb]"
```

## Commands

### `push`

Open an ephemeral W&B run and log the deterministic and MC-Dropout ONNX
artifacts produced by a training run.

| Flag | Description |
|---|---|
| `--det-path` | Path to the deterministic ONNX export. |
| `--mcd-path` | Path to the MC-Dropout ONNX export. |
| `--run-id` | W&B run ID to log the artifacts under. |
| `--det-collection` | Registry collection name for the deterministic artifact. |
| `--mcd-collection` | Registry collection name for the MC-Dropout artifact. |
| `--input-shape` | Model input tensor shape, repeatable (one value per dimension). |
| `--classes` | Ordered class labels the model predicts, repeatable. |

```bash
radiologist-registry push \
  --det-path model.onnx --mcd-path model_mcd.onnx --run-id abc123 \
  --det-collection det-models --mcd-collection mcd-models \
  --input-shape 1 --input-shape 3 --input-shape 224 --input-shape 224 \
  --classes NORMAL --classes PNEUMONIA
```

### `pull`

Resolve (if any selector flag is given) then download an artifact, or treat
`path` as a raw qualified artifact path.

| Argument / Flag | Description |
|---|---|
| `path` | Base artifact path, or a raw qualified artifact path when no selector flag is given. |
| `--local-dir` | Directory to download the artifact into. |
| `--run-id` | Resolve the artifact logged by this run directly. |
| `--tags` | Restrict the run search to these tag(s). |
| `--groups` | Restrict the run search to these group(s). |
| `--metric` | Summary metric used to rank candidate runs (highest first). |
| `--version` | Explicit version or alias to resolve. |
| `--include-sweeps` | Include sweep runs as eligible candidates. |

```bash
radiologist-registry pull entity/project --local-dir ./models --run-id abc123
```

### `resolve`

Resolve a selector to a qualified artifact name and version, without
downloading anything.

| Argument / Flag | Description |
|---|---|
| `path` | Base artifact path to resolve. |
| `--run-id` | Resolve the artifact logged by this run directly. |
| `--tags` | Restrict the run search to these tag(s). |
| `--groups` | Restrict the run search to these group(s). |
| `--metric` | Summary metric used to rank candidate runs (highest first). |
| `--version` | Explicit version or alias to resolve. |
| `--include-sweeps` | Include sweep runs as eligible candidates. |

```bash
radiologist-registry resolve entity/project --run-id abc123
```

### `promote`

Link a run's deterministic and MC-Dropout artifacts to their collections.
The shared alias is `production` unless either collection already has a
`production` member, in which case it becomes `staging`. Prompts for
confirmation unless `--force` is given.

| Argument / Flag | Description |
|---|---|
| `path` | Base artifact path shared by both artifacts. |
| `--run-id` | Run whose `best` artifacts should be promoted. |
| `--det-collection` | Collection to link the deterministic artifact to. |
| `--mcd-collection` | Collection to link the MC-Dropout artifact to. |
| `--force` | Skip the confirmation prompt. |

```bash
radiologist-registry promote entity/project --run-id abc123 \
  --det-collection det-models --mcd-collection mcd-models --force
```

### `transition-to-production`

Flip the `staging` member of each collection to `production`. Prompts for
confirmation unless `--force` is given. Raises if either collection has no
`staging` member.

| Flag | Description |
|---|---|
| `--det-collection` | Collection holding the deterministic artifact. |
| `--mcd-collection` | Collection holding the MC-Dropout artifact. |
| `--force` | Skip the confirmation prompt. |

```bash
radiologist-registry transition-to-production \
  --det-collection det-models --mcd-collection mcd-models --force
```

### `list`

List every member of a collection with its current aliases.

| Flag | Description |
|---|---|
| `--type` | Artifact type of the collection (e.g. `model`). |
| `--collection` | Name of the collection to list. |

```bash
radiologist-registry list --type model --collection det-models
```

### `alias`

Manage the alias list of a single artifact directly.

| Subcommand | Arguments | Description |
|---|---|---|
| `alias get <artifact_path>` | `artifact_path` — fully qualified artifact path | Print the artifact's current aliases. |
| `alias set <artifact_path> <alias>` | `artifact_path`, `alias` | Add an alias to the artifact. |
| `alias remove <artifact_path> <alias>` | `artifact_path`, `alias` | Remove an alias from the artifact. |

```bash
radiologist-registry alias set entity/project/model-abc123:best staging
radiologist-registry alias get entity/project/model-abc123:best
radiologist-registry alias remove entity/project/model-abc123:best staging
```

## Python API equivalent

Every command is a thin wrapper over `WandbRegistry`, so any workflow above
can also be scripted directly:

```python
from radiologist.registry import RegistrySelector, WandbRegistry, resolve_selector

registry = WandbRegistry()
selector = RegistrySelector(path="entity/project", run_id="abc123")
ref = resolve_selector(selector, registry)
local_ckpt = registry.download(ref, local_dir="./models")
```

See the [API Reference](api-registry.md) for the full `WandbRegistry`
method documentation.
