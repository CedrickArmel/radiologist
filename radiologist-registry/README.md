# radiologist-registry

[![ci](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml/badge.svg)](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CedrickArmel/radiologist/branch/main/graph/badge.svg)](https://codecov.io/gh/CedrickArmel/radiologist)
[![PyPI](https://img.shields.io/pypi/v/radiologist-registry)](https://pypi.org/project/radiologist-registry/)
![tested on](https://img.shields.io/badge/tested%20on-ubuntu--latest%20%7C%20python%203.10-blue)

W&B model registry facade — resolve, download, push, and promote ONNX/checkpoint
artifacts for the radiologist pipeline. Ships a `WandbRegistry` library and a
Typer CLI (`radiologist-registry`) built on top of it.

## Installation

### Hard dependencies (always installed)

```bash
pip install radiologist-registry
```

### Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `wandb` | `wandb` | `WandbRegistry` (all registry operations) |
| `cli` | `typer` | `radiologist-registry` CLI entry point |

```bash
pip install "radiologist-registry[cli,wandb]"
```

## Public API

Exported from `radiologist.registry`:

- **`ModelRegistry`** — `Protocol` describing the registry backend contract:
  `resolve`, `download`, `pull`, `log_model_artifacts`,
  `list_collection_artifacts`, `promote`, `transition_to_production`,
  `get_aliases`, `set_alias`, `remove_alias`.
- **`WandbRegistry`** — the W&B-backed implementation of `ModelRegistry`.
  Constructs eagerly (no injected dependencies needed for CLI use).
- **`RegistrySelector`** / **`resolve_selector`** — a declarative,
  framework-neutral description of which artifact to resolve
  (`path`, `run_id`, `groups`, `tags`, `metric`, `version`,
  `include_sweeps`). `resolve_selector(selector, registry)` validates that
  `run_id` and `tags` are not both set, then delegates to
  `registry.resolve(...)`.
- **`ArtifactRef`** — resolved artifact pointer (`qualified_name`, `run_id`,
  `artifact_name`, `version`).
- **`ExportResult`** — paths and metadata for a freshly exported model pair
  (`det_path`, `mcd_path`, `run_id`, `input_shape`, `classes`).
- **`LoggedArtifacts`** — qualified names of artifacts logged to an active run
  but not yet linked to a collection (`det_qualified_name`,
  `mcd_qualified_name`, `run_id`).
- **`PromoteResult`** — outcome of a link/transition transaction; the
  deterministic and MC-Dropout artifacts always share one `alias`
  (`det_qualified_name`, `mcd_qualified_name`, `alias`).
- **`CollectionMember`** — one artifact version in a collection together with
  its current alias list (`qualified_name`, `aliases`).

```python
from radiologist.registry import RegistrySelector, WandbRegistry, resolve_selector

registry = WandbRegistry()
selector = RegistrySelector(path="entity/project", run_id="abc123")
ref = resolve_selector(selector, registry)
local_ckpt = registry.download(ref, local_dir="./models")
```

## CLI reference

All commands construct a `WandbRegistry()` internally — no separate wiring
required. Errors surface as `Error: {message}` on stderr with a non-zero exit
code.

| Command | Flags | Description |
|---|---|---|
| `resolve <path>` | `--run-id`, `--tags`, `--groups`, `--metric`, `--version`, `--include-sweeps` | Resolve a selector to a qualified name and version. |
| `pull <path>` | `--local-dir`, `--run-id`, `--tags`, `--groups`, `--metric`, `--version`, `--include-sweeps` | Resolve (if any selector flag is given) then download, or treat `path` as a raw artifact path. |
| `push` | `--det-path`, `--mcd-path`, `--run-id`, `--det-collection`, `--mcd-collection`, `--input-shape` (repeatable), `--classes` (repeatable) | Open an ephemeral W&B run and log the deterministic and MC-Dropout artifacts. |
| `promote <path>` | `--run-id`, `--det-collection`, `--mcd-collection`, `--force` | Link both artifacts to their collections with `staging`/`production` alias; prompts unless `--force`. |
| `transition-to-production` | `--det-collection`, `--mcd-collection`, `--force` | Flip the `staging` member of each collection to `production`; prompts unless `--force`. |
| `list` | `--type`, `--collection` | List every member of a collection with its current aliases. |
| `alias get <artifact_path>` | — | Print the artifact's current aliases. |
| `alias set <artifact_path> <alias>` | — | Add an alias to the artifact. |
| `alias remove <artifact_path> <alias>` | — | Remove an alias from the artifact. |

### Examples

```bash
radiologist-registry resolve entity/project --run-id abc123

radiologist-registry pull entity/project --local-dir ./models --run-id abc123

radiologist-registry push \
  --det-path model.onnx --mcd-path model_mcd.onnx --run-id abc123 \
  --det-collection det-models --mcd-collection mcd-models \
  --input-shape 1 --input-shape 3 --input-shape 224 --input-shape 224 \
  --classes NORMAL --classes PNEUMONIA

radiologist-registry promote entity/project --run-id abc123 \
  --det-collection det-models --mcd-collection mcd-models --force

radiologist-registry transition-to-production \
  --det-collection det-models --mcd-collection mcd-models --force

radiologist-registry list --type model --collection det-models

radiologist-registry alias set entity/project/model-abc123:best staging
radiologist-registry alias get entity/project/model-abc123:best
radiologist-registry alias remove entity/project/model-abc123:best staging
```
