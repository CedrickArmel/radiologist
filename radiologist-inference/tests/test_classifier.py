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

"""Behavioral tests for Classifier / BasePredictor.

Tests drive through the public API only: Classifier.from_path,
Classifier.from_registry, Classifier.predict. Fixtures build real ONNX
models so no mocks are needed for locally owned code; the wandb SDK
boundary is exercised via a fake registry or the shared optional._wandb
sentinel.
"""

from typing import List, Tuple
from unittest.mock import patch

import numpy as np
import pytest
from _helpers import build_det_onnx

from radiologist.inference import BasePredictor, Classifier, Prediction

CLASSES = ["NORMAL", "ABNORMAL"]


class _FakeRegistry:
    def __init__(self, det_path: str) -> None:
        self._det_path = det_path
        self.calls: List[Tuple[str, str]] = []

    def pull(self, artifact_path: str, local_dir: str) -> str:
        self.calls.append((artifact_path, local_dir))
        return self._det_path


class TestClassifierFromPath:
    def test_from_path_returns_classifier_instance(self, det_onnx_path):
        """from_path called on Classifier must return a Classifier instance."""
        classifier = Classifier.from_path(det_path=det_onnx_path)
        assert isinstance(classifier, Classifier)
        assert isinstance(classifier, BasePredictor)


class TestClassifierPredict:
    def test_predict_returns_prediction_with_class_keys_and_argmax(self, det_onnx_path):
        """predict must return a Prediction whose probability keys equal the
        model classes and whose predicted_class is the argmax."""
        classifier = Classifier.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = classifier.predict(image=image)

        assert isinstance(result, Prediction)
        assert set(result.probabilities.keys()) == set(CLASSES)
        expected = max(result.probabilities, key=result.probabilities.__getitem__)
        assert result.predicted_class == expected
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5


class TestClassifierDeploymentPrior:
    def test_deployment_prior_changes_probabilities_and_stays_normalized(
        self, det_onnx_path
    ):
        """Supplying deployment_prior must change probabilities while the
        result still sums to ~1."""
        classifier = Classifier.from_path(det_path=det_onnx_path)
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        no_prior = classifier.predict(image=image)
        with_prior = classifier.predict(
            image=image, deployment_prior={"NORMAL": 0.9, "ABNORMAL": 0.1}
        )

        assert no_prior.probabilities != with_prior.probabilities
        assert abs(sum(with_prior.probabilities.values()) - 1.0) < 1e-5


class TestClassifierEmbeddedPrior:
    def test_embedded_training_prior_applied_when_no_deployment_prior(self, tmp_path):
        """When the model embeds a training_prior and no deployment_prior is
        given, the embedded prior must be applied."""
        onnx_with_prior = build_det_onnx(
            tmp_path,
            priors={"NORMAL": 0.7, "ABNORMAL": 0.3},
            filename="with_prior.onnx",
        )
        onnx_no_prior = build_det_onnx(tmp_path, priors=None, filename="no_prior.onnx")
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        with_embedded = Classifier.from_path(det_path=onnx_with_prior).predict(
            image=image
        )
        no_embedded = Classifier.from_path(det_path=onnx_no_prior).predict(image=image)

        assert with_embedded.probabilities != no_embedded.probabilities


class TestClassifierFromRegistry:
    def test_from_registry_with_injected_registry_matches_from_path(
        self, det_onnx_path, tmp_path
    ):
        """from_registry with an injected fake registry must return a
        Classifier whose predict yields the same class keys as from_path."""
        fake_registry = _FakeRegistry(det_onnx_path)

        registry_classifier = Classifier.from_registry(
            artifact_path="entity/project/name:v0",
            local_dir=str(tmp_path),
            registry=fake_registry,
        )
        path_classifier = Classifier.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        registry_result = registry_classifier.predict(image=image)
        path_result = path_classifier.predict(image=image)

        assert isinstance(registry_classifier, Classifier)
        assert set(registry_result.probabilities.keys()) == set(
            path_result.probabilities.keys()
        )
        assert registry_result.predicted_class == path_result.predicted_class
        assert fake_registry.calls == [("entity/project/name:v0", str(tmp_path))]

    def test_from_registry_raises_runtime_error_naming_registry_when_wandb_absent(
        self, tmp_path
    ):
        """from_registry with no injected registry and wandb absent must
        raise RuntimeError naming 'registry'."""
        import radiologist.registry.optional as optional_mod

        with patch.object(optional_mod, "_wandb", None):
            with pytest.raises(RuntimeError, match="registry"):
                Classifier.from_registry(
                    artifact_path="entity/project/name:v0",
                    local_dir=str(tmp_path),
                )
