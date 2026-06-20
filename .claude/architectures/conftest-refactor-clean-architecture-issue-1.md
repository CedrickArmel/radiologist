## 🦴 Conftest refactor — shared fixture seam (skeleton)

### Context

This issue establishes the entire shared-fixture surface the two slices depend on, in one shared seam. It is the only thing #2 and #3 block on. Concretely it: (1) creates the root `conftest.py` with the single `sys.path` shim; (2) reworks `radiologist-core/tests/conftest.py` to promote the four pure-value fixtures to `session` scope, merge the two transforms into one session-scoped `transform`, and add the composed function-scoped `dm` fixture; (3) reworks `radiologist-inference/tests/conftest.py` to own every canonical ONNX builder and expose explicit per-variant fixtures. Per the Clean Architecture strategy this also formalizes the test layout in `pyproject.toml`. The seam is a real, working set of fixtures (it must be — fixtures cannot be `NotImplementedError` stubs and still let slices run green), but it changes no behavioral test assertions. See the epic spec for the overall shape.

### Module layout

```
conftest.py                                   # NEW — root: sys.path shim only, applies to all 5 packages
radiologist-utils/tests/conftest.py           # EDIT — drop shim line (now inherited from root)
radiologist-registry/tests/conftest.py        # EDIT — drop shim line
radiologist-etl/tests/conftest.py             # EDIT — drop shim line, keep package fixtures
radiologist-core/tests/conftest.py            # EDIT — session scopes, merged transform, dm fixture
radiologist-inference/tests/conftest.py       # EDIT — canonical builders + explicit variant fixtures
pyproject.toml                                # EDIT — [tool.pytest.ini_options] testpaths/rootdir
```

### Interface contracts

<!-- Fixtures and builders are real implementations (a conftest cannot stub fixtures and stay
     importable). Signatures + contracts shown; bodies follow the existing originals verbatim
     except where a contract note states a change. -->

##### `conftest.py` (root, NEW)

```python
# contract: inserts each package's src/ onto sys.path for namespace-package test imports.
# Replaces the per-package shim in all 5 conftest files. pytest loads the nearest-rootdir
# conftest first, so this runs before any package conftest.
import sys
from pathlib import Path
# for each workspace member: sys.path.insert(0, str(member / "src"))
```

##### `radiologist-core/tests/conftest.py`

```python
@pytest.fixture(scope="session")
def label_map() -> dict:
    # contract: {"NORMAL": "normal", "ABNORMAL": "abnormal"}; immutable shared value

@pytest.fixture(scope="session")
def classes() -> list:
    # contract: ["abnormal", "normal"]; immutable shared value

@pytest.fixture(scope="session")
def batch_size() -> int:
    # contract: 2

@pytest.fixture(scope="session")
def transform() -> "T.Compose":
    # contract: T.Compose([T.Resize((8, 8)), T.ToTensor()]); the single merged transform
    # used for BOTH train and eval (originals were byte-identical)

@pytest.fixture()
def dm(shard_root, label_map, classes, transform, train_loader_partial,
       eval_loader_partial, split_manifest_uri, batch_size) -> "WebDatasetDataModule":
    # contract: returns a WebDatasetDataModule wired with the default common-case args
    # (transform passed as both train_transform and eval_transform). function-scoped because
    # shard_root/split_manifest_uri derive from tmp_path. Tests needing non-default kwargs
    # (shared_map, batch_size=1, etc.) do NOT use this fixture.
```

<!-- train_loader_partial, eval_loader_partial, split_manifest_uri, shard_root keep their
     existing definitions and scopes unchanged. train_transform/eval_transform are REMOVED. -->

##### `radiologist-inference/tests/conftest.py`

```python
def build_det_onnx(tmp_path, priors: Optional[dict] = None,
                   filename: str = "model_det.onnx",
                   feat_nonzero: bool = False) -> str:
    # contract: builds a deterministic 2-class ONNX classifier (Reshape->Gemm->Softmax->Identity).
    # feat_const = np.random.rand(...) when feat_nonzero else np.zeros(...). Embeds metadata;
    # embeds training_prior when priors given. Module-level function (test_app imports it).

def build_mcd_onnx(tmp_path, filename: str = "model_mcd.onnx") -> str:
    # contract: stochastic RandomUniform->Softmax model; logits vary per session.run. Module-level.

@pytest.fixture()
def det_onnx_path(tmp_path) -> str:
    # contract: build_det_onnx(tmp_path) — zeros feature map (default case)

@pytest.fixture()
def det_onnx_path_nonzero(tmp_path) -> str:
    # contract: build_det_onnx(tmp_path, feat_nonzero=True) — explicit Score-CAM variant,
    # nonzero feature maps. Clean Architecture: a named fixture, NOT a flag at call sites.

@pytest.fixture()
def mcd_onnx_path(tmp_path) -> str:
    # contract: build_mcd_onnx(tmp_path)

@pytest.fixture()
def predictor_with_mcd(tmp_path) -> "Predictor":
    # contract: Predictor.from_path(det_path=build_det_onnx(...), mcd_path=build_mcd_onnx(...))
    # promoted here from test_mc_dropout so any inference test can use it

@pytest.fixture()
def predictor_without_mcd(tmp_path) -> "Predictor":
    # contract: Predictor.from_path(det_path=build_det_onnx(...)) — no mcd

@pytest.fixture()
def sample_image() -> "np.ndarray":
    # contract: np.zeros((224, 224, 3), dtype=np.uint8)
```

##### `pyproject.toml`

```toml
[tool.pytest.ini_options]
# contract: formalize rootdir + testpaths so the root conftest is always the rootdir conftest
# and discovery covers all five packages' tests/ dirs.
```

### Acceptance criteria

- [ ] mypy clean; pytest green (no behavioral tests added — the existing suite stays green with the new fixtures in place and the per-package shim removed)
