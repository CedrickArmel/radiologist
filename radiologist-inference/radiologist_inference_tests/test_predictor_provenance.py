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

"""Behavioral tests: resolved artifact provenance travels with the predictor.

Drives through the public API only: BasePredictor.provenance (inherited by
Classifier), from_selector, from_path. Fixtures build a real ONNX model so no
mocks are needed for locally owned code; the registry is faked at the
resolve/pull boundary the same way test_classifier.py does.
"""

from typing import List, Optional, Tuple

import numpy as np
import pytest

from radiologist.inference import Classifier
from radiologist.registry import ArtifactRef, RegistrySelector

CLASSES = ["NORMAL", "ABNORMAL"]


class _FakeSelectorRegistry:
    def __init__(self, det_path: str) -> None:
        self._det_path = det_path
        self.resolve_calls: List[Tuple[str, Optional[str]]] = []
        self.pull_calls: List[Tuple[str, str]] = []

    def resolve(
        self,
        path: str,
        run_id=None,
        groups=None,
        tags=None,
        metric=None,
        version=None,
        include_sweeps: bool = False,
    ) -> ArtifactRef:
        self.resolve_calls.append((path, run_id))
        resolved_run_id = run_id or "resolved-run"
        return ArtifactRef(
            qualified_name=f"{path}/model-{resolved_run_id}:best",
            run_id=resolved_run_id,
            artifact_name=f"model-{resolved_run_id}",
            version="best",
        )

    def pull(self, artifact_path: str, local_dir: str) -> str:
        self.pull_calls.append((artifact_path, local_dir))
        return self._det_path


class TestPredictorProvenance:
    def test_from_selector_reports_fully_qualified_artifact_name(
        self, det_onnx_path, tmp_path
    ):
        """A predictor resolved from a registry selector reports the fully
        qualified name of the artifact it was resolved from, including the
        entity/project prefix."""
        fake_registry = _FakeSelectorRegistry(det_onnx_path)
        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        classifier = Classifier.from_selector(
            selector, local_dir=str(tmp_path), registry=fake_registry
        )

        assert classifier.provenance is not None
        assert (
            classifier.provenance.qualified_name
            == "entity/project/model/model-run123:best"
        )

    def test_from_selector_reports_resolved_artifact_version(
        self, det_onnx_path, tmp_path
    ):
        """A predictor resolved from a registry selector reports the resolved
        artifact version."""
        fake_registry = _FakeSelectorRegistry(det_onnx_path)
        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        classifier = Classifier.from_selector(
            selector, local_dir=str(tmp_path), registry=fake_registry
        )

        assert classifier.provenance is not None
        assert classifier.provenance.version == "best"

    def test_from_path_reports_no_provenance(self, det_onnx_path):
        """A predictor loaded from a local ONNX file path reports no
        provenance (provenance is None)."""
        classifier = Classifier.from_path(model_path=det_onnx_path)

        assert classifier.provenance is None

    def test_from_selector_prediction_matches_direct_path_load(
        self, det_onnx_path, tmp_path
    ):
        """A predictor resolved from a registry selector still produces the
        same prediction as the same model loaded directly from disk —
        provenance is additive; inference behavior is unaffected."""
        fake_registry = _FakeSelectorRegistry(det_onnx_path)
        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        selector_classifier = Classifier.from_selector(
            selector, local_dir=str(tmp_path), registry=fake_registry
        )
        path_classifier = Classifier.from_path(model_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        selector_result = selector_classifier.predict(image=image)
        path_result = path_classifier.predict(image=image)

        assert selector_result.predicted_class == path_result.predicted_class
        assert selector_result.probabilities == path_result.probabilities

    def test_provenance_cannot_be_reassigned_from_outside_the_predictor(
        self, det_onnx_path
    ):
        """The reported provenance cannot be reassigned from outside the
        predictor (read-only property)."""
        classifier = Classifier.from_path(model_path=det_onnx_path)

        with pytest.raises(AttributeError):
            classifier.provenance = ArtifactRef(
                qualified_name="entity/project/model/model-x:best",
                run_id="x",
                artifact_name="model-x",
                version="best",
            )
