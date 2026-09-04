# radiologist-registry

[![ci](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml/badge.svg)](https://github.com/CedrickArmel/radiologist/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/CedrickArmel/radiologist/branch/main/graph/badge.svg)](https://codecov.io/gh/CedrickArmel/radiologist)
[![PyPI](https://img.shields.io/pypi/v/radiologist-registry)](https://pypi.org/project/radiologist-registry/)
![tested on](https://img.shields.io/badge/tested%20on-ubuntu--latest%20%7C%20python%203.10-blue)

W&B model registry facade — resolve, download, push, and promote ONNX/checkpoint
artifacts for the radiologist pipeline. Ships the `WandbRegistry` library;
its CLI is exposed through the `registry` command group of the unified
`radiologist` CLI, see [radiologist-cli](../radiologist-cli/README.md).

## Installation

### Hard dependencies (always installed)

```bash
pip install radiologist-registry
```

### Optional extras

| Extra | Installs | Enables |
|---|---|---|
| `wandb` | `wandb` | `WandbRegistry` (all registry operations) |

```bash
pip install "radiologist-registry[wandb]"
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
  `run_id` and `tags` are not both set, and that a registry-backed selector
  (any of `run_id`/`tags`/`groups`/`metric`/`version` set) carries a
  non-blank `path` — the entity/project to resolve against — then delegates
  to `registry.resolve(...)`.
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

## CLI

This package no longer ships its own console script. Its commands are the
`registry` group of the unified `radiologist` CLI — install
`radiologist-cli[registry]` and see
[docs/reference/cli-registry.md](../docs/reference/cli-registry.md) for the
full command reference and examples (`radiologist registry resolve ...`,
`radiologist registry push ...`, etc.).
