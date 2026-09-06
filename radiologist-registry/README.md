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

## Using the public API

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
from radiologist.registry import (
    ArtifactRef, CollectionMember, ExportResult, LoggedArtifacts,
    ModelRegistry, PromoteResult, RegistrySelector, WandbRegistry,
    resolve_selector, selector_from_flags,
)
```

`WandbRegistry()` takes no constructor arguments — it reads W&B credentials from
the environment (see [docs/reference/environment.md](../docs/reference/environment.md)).

### Resolving and downloading a model

The `RegistrySelector` → `resolve_selector` → `download` path is the one the
inference CLI uses. Resolution and download are separate steps so a caller can
inspect the chosen `ArtifactRef` (and record its provenance) before paying for
the download:

```python
from radiologist.registry import RegistrySelector, WandbRegistry, resolve_selector

registry = WandbRegistry()

selector = RegistrySelector(
    path="entity/project/chest-xray-det",
    tags=["production"],
    metric="val_score",
)
ref = resolve_selector(selector, registry)
# ArtifactRef(qualified_name=..., run_id=..., artifact_name=..., version=...)

local_path = registry.download(ref, local_dir="/tmp/models")
```

`resolve_selector` raises `ValueError` when both `run_id` and `tags` are given
("Provide either --run-id or --tags, not both."), and when a registry-backed
selector has a blank `path`.

Building a selector from flag-shaped values — note this helper's parameter order
is `path, run_id, tags, groups, ...`, which swaps `tags` and `groups` relative
to `RegistrySelector`'s own field order:

```python
from radiologist.registry import selector_from_flags

selector = selector_from_flags(
    path="entity/project/chest-xray-det",
    tags=["production"],
    metric="val_score",
)
selector.is_registry_backed()   # False when only `path` is set -> treat as a local path
```

For a fully-qualified artifact path you already know, skip resolution entirely:

```python
local_path = registry.pull("entity/project/model-abc123:best", local_dir="/tmp/models")
```

### Publishing and promoting

`log_model_artifacts` uploads the two ONNX files produced by
`radiologist.core.registry.export_onnx` against an active W&B run; `promote`
then links them into the two registry collections:

```python
logged = registry.log_model_artifacts(
    export_result=export_result,   # radiologist.registry.ExportResult
    run=wandb_run,
    ckpt_path="/path/to/best.ckpt",
    last_ckpt_path="/path/to/last.ckpt",
)

promoted = registry.promote(
    path="entity/project",
    run_id=export_result.run_id,
    det_collection="chest-xray-det",
    mcd_collection="chest-xray-mcd",
)
promoted.alias   # "staging" or "production"
```

**`promote` never overwrites a live model.** It resolves `run_id` and
`{run_id}-mcd` at version `best`, then assigns `alias = "staging"` when a
`production`-aliased member already exists in the collection, and `"production"`
only when none does. Cutting a staged model over is a separate, explicit step:

```python
registry.transition_to_production(
    det_collection="chest-xray-det",
    mcd_collection="chest-xray-mcd",
)
```

### Inspecting a collection and managing aliases

```python
for member in registry.list_collection_artifacts("model", "chest-xray-det"):
    print(member.qualified_name, member.aliases)

registry.get_aliases("entity/project/model-abc123:v3")
registry.set_alias("entity/project/model-abc123:v3", "candidate")
registry.remove_alias("entity/project/model-abc123:v3", "candidate")
```

### Result dataclasses

All are frozen value objects — no behavior, safe to log and serialize.

| Type | Fields |
|---|---|
| `ArtifactRef` | `qualified_name`, `run_id`, `artifact_name`, `version` |
| `ExportResult` | `det_path`, `mcd_path`, `run_id`, `input_shape`, `classes` |
| `LoggedArtifacts` | `det_qualified_name`, `mcd_qualified_name`, `run_id` |
| `PromoteResult` | `det_qualified_name`, `mcd_qualified_name`, `alias` |
| `CollectionMember` | `qualified_name`, `aliases` |
| `RegistrySelector` | `path`, `run_id`, `groups`, `tags`, `metric`, `version`, `include_sweeps` |

## Extending the registry backend

**There are no Hydra config groups in this package** — it ships no yaml, no
`conf/` directory, and `pyproject.toml` declares no dependencies at all, let
alone Hydra. Extension here happens through a Python protocol, not configuration.

`ModelRegistry` (`radiologist.registry.interface`) is the pluggable seam. It is a
plain `typing.Protocol`, so your backend is **structurally** typed: implement the
methods and pass the instance — no base class, no registration, no import of
ours in your class definition.

```python
from typing import Any, List, Optional, Union

from radiologist.registry import (
    ArtifactRef, CollectionMember, ExportResult, LoggedArtifacts, PromoteResult,
)


class LocalDirRegistry:
    """A ModelRegistry backed by a directory on disk."""

    def resolve(
        self,
        path: str,
        run_id: Optional[str] = None,
        groups: Optional[Union[str, List[str]]] = None,
        tags: Optional[Union[str, List[str]]] = None,
        metric: Optional[str] = None,
        version: Optional[str] = None,
        include_sweeps: bool = False,
    ) -> ArtifactRef: ...

    def download(self, ref: ArtifactRef, local_dir: str) -> str: ...
    def pull(self, artifact_path: str, local_dir: str) -> str: ...

    def log_model_artifacts(
        self,
        export_result: ExportResult,
        run: Any,
        ckpt_path: str,
        last_ckpt_path: Optional[str] = None,
    ) -> LoggedArtifacts: ...

    def list_collection_artifacts(
        self, type_name: str, collection_name: str
    ) -> List[CollectionMember]: ...

    def promote(
        self, path: str, run_id: str, det_collection: str, mcd_collection: str
    ) -> PromoteResult: ...

    def transition_to_production(
        self, det_collection: str, mcd_collection: str
    ) -> PromoteResult: ...

    def get_aliases(self, artifact_path: str) -> List[str]: ...
    def set_alias(self, artifact_path: str, alias: str) -> None: ...
    def remove_alias(self, artifact_path: str, alias: str) -> None: ...
```

Your instance is accepted anywhere `WandbRegistry` is:

```python
from radiologist.inference import Classifier
from radiologist.registry import resolve_selector

registry = LocalDirRegistry()
ref = resolve_selector(selector, registry)
predictor = Classifier.from_selector(selector, local_dir="/tmp/models", registry=registry)
```

Two details to be aware of:

- `ModelRegistry` is **not** `@runtime_checkable`, so conformance is checked by
  mypy at type-check time, not by `isinstance` at run time. A partial
  implementation fails at the first missing attribute access.
- The protocol's `resolve` accepts `Optional[Union[str, List[str]]]` for
  `groups`/`tags`, while `RegistrySelector` types both as `Optional[List[str]]`.
  Accept the wider form in your implementation.

`WandbRegistry`'s own four internal collaborators (`_WandbResolver`,
`_WandbUploader`, `_WandbAliasManager`, `_WandbCollectionLister`) are
underscore-private and constructed eagerly in `__init__` — they are not
injectable and are not an extension point. Implement `ModelRegistry` instead.

## CLI

This package no longer ships its own console script. Its commands are the
`registry` group of the unified `radiologist` CLI — install
`radiologist-cli[registry]` and see
[docs/reference/cli-registry.md](../docs/reference/cli-registry.md) for the
full command reference and examples (`radiologist registry resolve ...`,
`radiologist registry push ...`, etc.).
