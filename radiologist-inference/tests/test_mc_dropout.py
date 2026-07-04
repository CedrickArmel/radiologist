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

"""Behavioral tests for MCDropoutPredictor.predict_with_uncertainty and the
stateless mc_dropout_predict helper.

Tests drive through the public API only. Fixtures build real ONNX models so
no mocks are needed for local code.
"""

from typing import List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pytest
from _helpers import build_det_onnx, build_mcd_onnx

from radiologist.inference import MCDropoutPredictor
from radiologist.registry import ArtifactRef, RegistrySelector

CLASSES = ["NORMAL", "ABNORMAL"]


class _FakeMcdSelectorRegistry:
    """Fake registry resolving det (run_id) vs mcd ({run_id}-mcd) artifacts.

    When run_id is None (tags/groups/metric-based selection), the same
    artifact path is returned for both resolutions, matching today's
    fallback semantics.
    """

    def __init__(self, det_path: str, mcd_path: str) -> None:
        self._det_path = det_path
        self._mcd_path = mcd_path
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
        if artifact_path.endswith("-mcd:best"):
            return self._mcd_path
        return self._det_path


# ---------------------------------------------------------------------------
# Tests: MCDropoutPredictor.predict_with_uncertainty
# ---------------------------------------------------------------------------


class TestPredictWithUncertainty:
    def test_returns_uncertainty_result_type(self, predictor_with_mcd, sample_image):
        """predict_with_uncertainty must return an UncertaintyResult."""
        from radiologist.inference.models import UncertaintyResult

        result = predictor_with_mcd.predict_with_uncertainty(sample_image, n_passes=10)
        assert isinstance(result, UncertaintyResult)

    def test_mean_probabilities_sum_to_one(self, predictor_with_mcd, sample_image):
        """mean_probabilities must sum to 1.0 within floating tolerance."""
        result = predictor_with_mcd.predict_with_uncertainty(sample_image, n_passes=10)
        total = sum(result.mean_probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_mean_probabilities_keyed_by_class_names(
        self, predictor_with_mcd, sample_image
    ):
        """mean_probabilities keys must match class names from model metadata."""
        result = predictor_with_mcd.predict_with_uncertainty(sample_image, n_passes=10)
        assert set(result.mean_probabilities.keys()) == set(CLASSES)

    def test_std_per_class_is_nonzero(self, predictor_with_mcd, sample_image):
        """std_per_class must be non-zero across stochastic passes."""
        result = predictor_with_mcd.predict_with_uncertainty(sample_image, n_passes=30)
        assert any(v > 0 for v in result.std_per_class.values())

    def test_predictive_entropy_is_nonnegative(self, predictor_with_mcd, sample_image):
        """predictive_entropy must be >= 0."""
        result = predictor_with_mcd.predict_with_uncertainty(sample_image, n_passes=10)
        assert result.predictive_entropy >= 0.0

    def test_n_passes_reflects_requested_count(self, predictor_with_mcd, sample_image):
        """UncertaintyResult.n_passes must equal the requested number of passes."""
        result = predictor_with_mcd.predict_with_uncertainty(sample_image, n_passes=15)
        assert result.n_passes == 15

    def test_larger_n_passes_reflected_in_result(
        self, predictor_with_mcd, sample_image
    ):
        """Calling with n_passes=50 must report n_passes=50."""
        result = predictor_with_mcd.predict_with_uncertainty(sample_image, n_passes=50)
        assert result.n_passes == 50

    def test_raises_runtime_error_when_no_mcd_model(
        self, predictor_without_mcd, sample_image
    ):
        """predict_with_uncertainty on a predictor loaded without mcd_path must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="MC-Dropout"):
            predictor_without_mcd.predict_with_uncertainty(sample_image)


# ---------------------------------------------------------------------------
# Tests: mc_dropout_predict (public API function)
# ---------------------------------------------------------------------------


class TestMcDropoutPredict:
    def test_returns_uncertainty_result(self, mcd_onnx_path, sample_image):
        """mc_dropout_predict must return an UncertaintyResult."""
        import onnxruntime as ort

        from radiologist.inference.mc_dropout import mc_dropout_predict
        from radiologist.inference.models import UncertaintyResult

        session = ort.InferenceSession(mcd_onnx_path)
        preprocessed = sample_image.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        result = mc_dropout_predict(session, preprocessed, n_passes=10)
        assert isinstance(result, UncertaintyResult)

    def test_mean_probs_sum_to_one(self, mcd_onnx_path, sample_image):
        """mc_dropout_predict mean_probabilities must sum to 1.0."""
        import onnxruntime as ort

        from radiologist.inference.mc_dropout import mc_dropout_predict

        session = ort.InferenceSession(mcd_onnx_path)
        preprocessed = sample_image.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        result = mc_dropout_predict(session, preprocessed, n_passes=10)
        total = sum(result.mean_probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_n_passes_recorded_in_result(self, mcd_onnx_path, sample_image):
        """mc_dropout_predict result must record the requested number of passes."""
        import onnxruntime as ort

        from radiologist.inference.mc_dropout import mc_dropout_predict

        session = ort.InferenceSession(mcd_onnx_path)
        preprocessed = sample_image.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        result = mc_dropout_predict(session, preprocessed, n_passes=20)
        assert result.n_passes == 20


# ---------------------------------------------------------------------------
# Tests: MCDropoutPredictor.from_selector
# ---------------------------------------------------------------------------


class TestMCDropoutFromSelector:
    def test_from_selector_resolves_det_and_mcd_suffixed_run_id(self, tmp_path):
        """from_selector with a run_id must resolve both the det artifact
        (run_id) and the mcd artifact ({run_id}-mcd), threading mean/std/
        input_shape into the loaded predictor (bugfix #139)."""
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        fake_registry = _FakeMcdSelectorRegistry(det_path, mcd_path)
        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        predictor = MCDropoutPredictor.from_selector(
            selector,
            local_dir=str(tmp_path),
            registry=fake_registry,
            mean=128.0,
            std=65.0,
        )

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict_with_uncertainty(image, n_passes=10)

        assert isinstance(predictor, MCDropoutPredictor)
        assert set(result.mean_probabilities.keys()) == set(CLASSES)
        assert abs(sum(result.mean_probabilities.values()) - 1.0) < 1e-5
        assert fake_registry.resolve_calls == [
            ("entity/project/model", "run123"),
            ("entity/project/model", "run123-mcd"),
        ]
        assert fake_registry.pull_calls == [
            ("entity/project/model/model-run123:best", str(tmp_path)),
            ("entity/project/model/model-run123-mcd:best", str(tmp_path)),
        ]

    def test_from_selector_mean_std_changes_normalization(self, tmp_path):
        """mean/std supplied to from_selector must actually change the
        normalization applied before MC-Dropout inference."""
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")

        default_predictor = MCDropoutPredictor.from_selector(
            RegistrySelector(path="entity/project/model", run_id="run123"),
            local_dir=str(tmp_path),
            registry=_FakeMcdSelectorRegistry(det_path, mcd_path),
        )
        normalized_predictor = MCDropoutPredictor.from_selector(
            RegistrySelector(path="entity/project/model", run_id="run123"),
            local_dir=str(tmp_path),
            registry=_FakeMcdSelectorRegistry(det_path, mcd_path),
            mean=128.0,
            std=65.0,
        )

        assert default_predictor._state.mean is None
        assert normalized_predictor._state.mean == 128.0
        assert normalized_predictor._state.std == 65.0

    def test_from_selector_without_run_id_reuses_same_selector_for_det_and_mcd(
        self, tmp_path
    ):
        """When selector.run_id is None (tags/groups/metric-based selection),
        the same selector must be reused for both det and mcd resolution —
        no -mcd suffix — matching today's CLI fallback behavior."""
        det_path = build_det_onnx(tmp_path, filename="det.onnx")
        mcd_path = build_mcd_onnx(tmp_path, filename="mcd.onnx")
        fake_registry = _FakeMcdSelectorRegistry(det_path, mcd_path)
        selector = RegistrySelector(path="entity/project/model", tags=["a", "b"])

        predictor = MCDropoutPredictor.from_selector(
            selector, local_dir=str(tmp_path), registry=fake_registry
        )

        assert isinstance(predictor, MCDropoutPredictor)
        assert fake_registry.resolve_calls == [
            ("entity/project/model", None),
            ("entity/project/model", None),
        ]

    def test_from_selector_raises_runtime_error_naming_inference_extra_when_wandb_absent(
        self, tmp_path
    ):
        """from_selector with no injected registry and wandb absent must
        raise RuntimeError naming radiologist-inference[registry]."""
        import radiologist.registry.optional as optional_mod

        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        with patch.object(optional_mod, "_wandb", None):
            with pytest.raises(
                RuntimeError, match=r"radiologist-inference\[registry\]"
            ):
                MCDropoutPredictor.from_selector(selector, local_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Tests: MCDropoutPredictor.from_path mean/std eager validation
# ---------------------------------------------------------------------------


class TestMCDropoutFromPathMeanStdValidation:
    def test_from_path_with_only_mean_raises_value_error(
        self, det_onnx_path, mcd_onnx_path
    ):
        """Supplying mean without std must raise ValueError eagerly at
        from_path, before any prediction is requested (bugfix review finding
        on PR #131 — MCDropoutPredictor delegates to the shared
        BasePredictor.from_path code path)."""
        with pytest.raises(ValueError, match="mean and std"):
            MCDropoutPredictor.from_path(
                det_path=det_onnx_path, mcd_path=mcd_onnx_path, mean=128.0
            )

    def test_from_selector_with_mismatched_mean_std_raises_before_any_pull(
        self, det_onnx_path, mcd_onnx_path, tmp_path
    ):
        """MCDropoutPredictor.from_selector has its own det+mcd resolve/pull
        sequence (it does not delegate resolution to BasePredictor); a
        mismatched mean/std pair must raise before either artifact is
        resolved or pulled (bugfix review finding on PR #131)."""
        fake_registry = _FakeMcdSelectorRegistry(det_onnx_path, mcd_onnx_path)
        selector = RegistrySelector(path="entity/project/model", run_id="run123")

        with pytest.raises(ValueError, match="mean and std"):
            MCDropoutPredictor.from_selector(
                selector,
                local_dir=str(tmp_path),
                registry=fake_registry,
                mean=128.0,
            )

        assert fake_registry.resolve_calls == []
        assert fake_registry.pull_calls == []
