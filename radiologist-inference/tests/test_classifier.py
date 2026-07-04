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

from typing import List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest
from _helpers import build_det_onnx

from radiologist.inference import BasePredictor, Classifier, Prediction
from radiologist.registry import ArtifactRef, RegistrySelector

CLASSES = ["NORMAL", "ABNORMAL"]


class _FakeRegistry:
    def __init__(self, det_path: str) -> None:
        self._det_path = det_path
        self.calls: List[Tuple[str, str]] = []

    def pull(self, artifact_path: str, local_dir: str) -> str:
        self.calls.append((artifact_path, local_dir))
        return self._det_path


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

    def test_from_registry_names_inference_extra_not_registry_extra(self, tmp_path):
        """The wandb-missing message must name radiologist-inference[registry],
        not radiologist-registry[wandb] (bugfix b)."""
        import radiologist.registry.optional as optional_mod

        with patch.object(optional_mod, "_wandb", None):
            with pytest.raises(
                RuntimeError, match=r"radiologist-inference\[registry\]"
            ):
                Classifier.from_registry(
                    artifact_path="entity/project/name:v0",
                    local_dir=str(tmp_path),
                )


class TestClassifierMeanStdNormalization:
    def test_predict_with_mean_and_std_differs_from_default(self, det_onnx_path):
        """Supplying mean/std must change probabilities relative to the
        default /255.0-only normalization."""
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        default_classifier = Classifier.from_path(det_path=det_onnx_path)
        normalized_classifier = Classifier.from_path(
            det_path=det_onnx_path, mean=128.0, std=65.0
        )

        default_result = default_classifier.predict(image=image)
        normalized_result = normalized_classifier.predict(image=image)

        assert default_result.probabilities != normalized_result.probabilities

    def test_from_path_with_only_mean_raises_value_error(self, det_onnx_path):
        """Supplying mean without std must raise ValueError eagerly at
        from_path, before any prediction is requested (fail-fast, bugfix
        review finding on PR #131)."""
        with pytest.raises(ValueError, match="mean and std"):
            Classifier.from_path(det_path=det_onnx_path, mean=128.0)

    def test_from_path_with_only_std_raises_value_error(self, det_onnx_path):
        """Supplying std without mean must raise ValueError eagerly at
        from_path, before any prediction is requested (fail-fast, bugfix
        review finding on PR #131)."""
        with pytest.raises(ValueError, match="mean and std"):
            Classifier.from_path(det_path=det_onnx_path, std=65.0)


class TestFromPathInputShapeFallback:
    def test_from_path_without_input_shape_metadata_raises_clear_error(self, tmp_path):
        """A model with no input_shape metadata and no default given must
        raise a clear ValueError."""
        det_path = build_det_onnx(
            tmp_path, filename="no_shape.onnx", omit_keys=["input_shape"]
        )

        with pytest.raises(ValueError, match="input_shape"):
            Classifier.from_path(det_path=det_path)

    def test_from_path_without_input_shape_metadata_succeeds_with_default(
        self, tmp_path
    ):
        """Passing input_shape explicitly must fill in for missing metadata."""
        det_path = build_det_onnx(
            tmp_path, filename="no_shape.onnx", omit_keys=["input_shape"]
        )

        classifier = Classifier.from_path(
            det_path=det_path, input_shape=[1, 3, 224, 224]
        )
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = classifier.predict(image=image)

        assert isinstance(result, Prediction)
        assert set(result.probabilities.keys()) == set(CLASSES)


class TestClassifierFromSelector:
    def test_from_selector_with_injected_registry_resolves_and_pulls(
        self, det_onnx_path, tmp_path
    ):
        """from_selector must resolve the selector then pull the resolved
        artifact, returning a Classifier equivalent to from_path."""
        fake_registry = _FakeSelectorRegistry(det_onnx_path)
        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        classifier = Classifier.from_selector(
            selector, local_dir=str(tmp_path), registry=fake_registry
        )
        path_classifier = Classifier.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        selector_result = classifier.predict(image=image)
        path_result = path_classifier.predict(image=image)

        assert isinstance(classifier, Classifier)
        assert selector_result.predicted_class == path_result.predicted_class
        assert fake_registry.resolve_calls == [("entity/project/model", "run123")]
        assert fake_registry.pull_calls == [
            ("entity/project/model/model-run123:best", str(tmp_path))
        ]

    def test_from_selector_raises_runtime_error_naming_inference_extra_when_wandb_absent(
        self, tmp_path
    ):
        """from_selector with no injected registry and wandb absent must raise
        RuntimeError naming radiologist-inference[registry] (bugfix b)."""
        import radiologist.registry.optional as optional_mod

        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        with patch.object(optional_mod, "_wandb", None):
            with pytest.raises(
                RuntimeError, match=r"radiologist-inference\[registry\]"
            ):
                Classifier.from_selector(selector, local_dir=str(tmp_path))

    def test_from_selector_with_mean_std_matches_from_path_normalization(
        self, det_onnx_path, tmp_path
    ):
        """from_selector must forward mean/std to from_path so a
        registry-backed load normalizes identically to a local-path load with
        the same params (bugfix #139)."""
        fake_registry = _FakeSelectorRegistry(det_onnx_path)
        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        selector_classifier = Classifier.from_selector(
            selector,
            local_dir=str(tmp_path),
            registry=fake_registry,
            mean=128.0,
            std=65.0,
        )
        path_classifier = Classifier.from_path(
            det_path=det_onnx_path, mean=128.0, std=65.0
        )
        default_classifier = Classifier.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        selector_result = selector_classifier.predict(image=image)
        path_result = path_classifier.predict(image=image)
        default_result = default_classifier.predict(image=image)

        assert selector_result.probabilities == path_result.probabilities
        assert selector_result.probabilities != default_result.probabilities

    def test_from_selector_with_mismatched_mean_std_raises_before_any_pull(
        self, det_onnx_path, tmp_path
    ):
        """A mismatched mean/std pair must raise ValueError before the
        selector is resolved or the artifact pulled — no wasted network I/O
        on a request that's doomed to fail (bugfix review finding on PR
        #131)."""
        fake_registry = _FakeSelectorRegistry(det_onnx_path)
        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        with pytest.raises(ValueError, match="mean and std"):
            Classifier.from_selector(
                selector,
                local_dir=str(tmp_path),
                registry=fake_registry,
                mean=128.0,
            )

        assert fake_registry.resolve_calls == []
        assert fake_registry.pull_calls == []
