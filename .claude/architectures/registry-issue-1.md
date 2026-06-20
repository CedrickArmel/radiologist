## 🦴 radiologist-registry — skeleton

### Context

Stubs the entire epic surface so every slice can start in parallel, blocked only on this. Creates the new `radiologist-registry` workspace package, the four domain dataclasses, the `ModelRegistry` Protocol, the `WandbRegistry` shell, and the core-side `export_onnx` signature. No behavior — type-checked contracts only. See `registry-spec.md`.

### Module layout

```
radiologist-registry/
├── pyproject.toml                 # member; module-name = "radiocovid.registry"; extra: registry=[wandb]
├── src/radiologist/registry/
│   ├── models.py                  # ArtifactRef, ExportResult, PromoteResult, AliasOp (frozen dataclasses)
│   ├── interface.py               # ModelRegistry Protocol
│   ├── resolver.py                # _resolve(...) -> ArtifactRef  (internal seam)
│   ├── uploader.py                # _upload_and_link(...) -> PromoteResult  (internal seam)
│   ├── alias_manager.py           # _set/_remove/_get aliases  (internal seam)
│   ├── wandb_registry.py          # WandbRegistry(ModelRegistry) facade
│   └── __init__.py                # public exports + __all__
└── tests/conftest.py              # sys.path shim + fake-wandb-api fixtures
radiologist-core/src/radiologist/core/registry/
└── export.py                      # export_onnx(...) -> ExportResult  (new; replaces promote.py export half)
```

### Interface contracts

##### `radiologist-registry/src/radiologist/registry/models.py`

```python
@dataclass(frozen=True)
class ArtifactRef:
    # contract: a fully-resolved pointer to one W&B artifact version
    qualified_name: str            # "entity/project/name:version"
    run_id: str
    aliases: Tuple[str, ...]

@dataclass(frozen=True)
class ExportResult:
    # contract: pure carrier across core->registry boundary; no torch/wandb refs
    run_id: str
    det_path: str                  # deterministic ONNX file path
    mcd_path: str                  # MC-dropout ONNX file path
    classes: Tuple[str, ...]
    input_shape: Tuple[int, ...]
    cam_target_layer: str
    metadata: Dict[str, str]       # extra onnx metadata_props already embedded

@dataclass(frozen=True)
class PromoteResult:
    # contract: result of uploading+linking an ExportResult to the registry
    det_qualified_name: str
    mcd_qualified_name: str
    collection: str
    aliases: Tuple[str, ...]

@dataclass(frozen=True)
class AliasOp:
    # contract: records one alias transition on an artifact version
    qualified_name: str
    added: Tuple[str, ...]
    removed: Tuple[str, ...]
```

##### `radiologist-registry/src/radiologist/registry/interface.py`

```python
class ModelRegistry(Protocol):
    def resolve(
        self,
        path: str,
        run_id: Optional[str] = None,
        groups: Optional[Union[str, List[str]]] = None,
        tags: Optional[Union[str, List[str]]] = None,
        metric: Optional[str] = None,
        version: Optional[str] = None,
        include_sweeps: bool = False,
    ) -> ArtifactRef:
        # contract: resolve a query to one ArtifactRef; raises LookupError if none match
        ...

    def download(self, ref: ArtifactRef, local_dir: str) -> str:
        # contract: download the artifact, return the local .ckpt path; FileNotFoundError if absent
        ...

    def promote(
        self, result: ExportResult, collection: str, aliases: Sequence[str]
    ) -> PromoteResult:
        # contract: upload det+mcd ONNX from result, link to collection with aliases
        ...

    def get_aliases(self, qualified_name: str) -> Tuple[str, ...]:
        # contract: current aliases on the artifact version
        ...

    def set_alias(self, qualified_name: str, alias: str) -> AliasOp:
        # contract: add alias (idempotent); returns the transition
        ...

    def remove_alias(self, qualified_name: str, alias: str) -> AliasOp:
        # contract: remove alias (idempotent); returns the transition
        ...
```

##### `radiologist-registry/src/radiologist/registry/wandb_registry.py`

```python
class WandbRegistry:
    # contract: the only ModelRegistry impl; raises RuntimeError at construction if wandb absent
    def __init__(self) -> None:
        raise NotImplementedError
    # all Protocol methods present, each: raise NotImplementedError
```

##### `radiologist-core/src/radiologist/core/registry/export.py`

```python
def export_onnx(
    ckpt_path: str,
    input_shape: Tuple[int, ...],
    classes: List[str],
    cam_target_layer: str,
    local_dir: str,
    run_id: str,
    precision: str,
    opset: int = 18,
) -> "ExportResult":
    # contract: load LModule from ckpt, export det+mcd ONNX into local_dir, embed metadata,
    #           return ExportResult; raises RuntimeError if onnx missing, AttributeError if layer absent.
    #           Imports ExportResult from radiologist.registry.models. No wandb.
    raise NotImplementedError
```

### Acceptance criteria

- [ ] `radiologist-registry` is a resolvable workspace member; `import radiologist.registry` and `from radiologist.registry import ModelRegistry, WandbRegistry, ArtifactRef, ExportResult, PromoteResult, AliasOp` succeed.
- [ ] mypy clean; pytest green (no behavioral tests — stubs typecheck and the existing suite stays green).
