# MIT License
#
# Copyright (c) 2026 @CedrickArmel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Any, List, Optional, Union

import pytest

from radiologist.registry import (
    ArtifactRef,
    RegistrySelector,
    resolve_selector,
    selector_from_flags,
)


class FakeModelRegistry:
    """Owned in-memory ModelRegistry stub — records args, returns a canned ref."""

    def __init__(self, ref: ArtifactRef) -> None:
        self.ref = ref
        self.calls: List[dict] = []

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
        self.calls.append(
            {
                "path": path,
                "run_id": run_id,
                "groups": groups,
                "tags": tags,
                "metric": metric,
                "version": version,
                "include_sweeps": include_sweeps,
            }
        )
        return self.ref

    def download(self, ref: ArtifactRef, local_dir: str) -> str:
        raise NotImplementedError

    def pull(self, artifact_path: str, local_dir: str) -> str:
        raise NotImplementedError

    def log_model_artifacts(
        self,
        export_result: Any,
        run: Any,
        ckpt_path: str,
        last_ckpt_path: Optional[str] = None,
    ) -> Any:
        raise NotImplementedError

    def list_collection_artifacts(self, type_name: str, collection_name: str) -> Any:
        raise NotImplementedError

    def promote(
        self, path: str, run_id: str, det_collection: str, mcd_collection: str
    ) -> Any:
        raise NotImplementedError

    def transition_to_production(self, det_collection: str, mcd_collection: str) -> Any:
        raise NotImplementedError

    def get_aliases(self, artifact_path: str) -> List[str]:
        raise NotImplementedError

    def set_alias(self, artifact_path: str, alias: str) -> None:
        raise NotImplementedError

    def remove_alias(self, artifact_path: str, alias: str) -> None:
        raise NotImplementedError


@pytest.fixture
def canned_ref() -> ArtifactRef:
    return ArtifactRef(
        qualified_name="entity/project/model:v1",
        run_id="run-123",
        artifact_name="model",
        version="v1",
    )


@pytest.fixture
def registry(canned_ref: ArtifactRef) -> FakeModelRegistry:
    return FakeModelRegistry(canned_ref)


def test_is_registry_backed_false_when_only_path_set() -> None:
    selector = RegistrySelector(path="models/local.onnx")
    assert selector.is_registry_backed() is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": "run-123"},
        {"tags": ["prod"]},
        {"groups": ["group-a"]},
        {"metric": "val_f1"},
        {"version": "v2"},
    ],
)
def test_is_registry_backed_true_when_registry_field_set(kwargs: dict) -> None:
    selector = RegistrySelector(path="entity/project/model", **kwargs)
    assert selector.is_registry_backed() is True


def test_resolve_selector_with_run_id_returns_ref_and_forwards_run_id(
    registry: FakeModelRegistry, canned_ref: ArtifactRef
) -> None:
    selector = RegistrySelector(path="entity/project/model", run_id="run-123")

    result = resolve_selector(selector, registry)

    assert result == canned_ref
    assert registry.calls[0]["run_id"] == "run-123"


def test_resolve_selector_with_tags_forwards_tags_as_list(
    registry: FakeModelRegistry, canned_ref: ArtifactRef
) -> None:
    selector = RegistrySelector(path="entity/project/model", tags=["prod", "best"])

    result = resolve_selector(selector, registry)

    assert result == canned_ref
    assert registry.calls[0]["tags"] == ["prod", "best"]
    assert isinstance(registry.calls[0]["tags"], list)


def test_resolve_selector_with_only_path_returns_ref_for_raw_path(
    registry: FakeModelRegistry, canned_ref: ArtifactRef
) -> None:
    selector = RegistrySelector(path="models/local.onnx")

    result = resolve_selector(selector, registry)

    assert result == canned_ref
    assert registry.calls[0]["path"] == "models/local.onnx"


def test_selector_from_flags_builds_equivalent_registry_selector() -> None:
    selector = selector_from_flags(
        path="entity/project/model",
        run_id="run-123",
        tags=["prod"],
        groups=["group-a"],
        metric="val_f1",
        version="v2",
        include_sweeps=True,
    )

    assert selector == RegistrySelector(
        path="entity/project/model",
        run_id="run-123",
        tags=["prod"],
        groups=["group-a"],
        metric="val_f1",
        version="v2",
        include_sweeps=True,
    )


def test_selector_from_flags_defaults_match_registry_selector_defaults() -> None:
    selector = selector_from_flags(path="models/local.onnx")

    assert selector == RegistrySelector(path="models/local.onnx")


def test_resolve_selector_raises_valueerror_when_run_id_and_tags_both_set(
    registry: FakeModelRegistry,
) -> None:
    selector = RegistrySelector(
        path="entity/project/model", run_id="run-123", tags=["prod"]
    )

    with pytest.raises(ValueError):
        resolve_selector(selector, registry)

    assert registry.calls == []
