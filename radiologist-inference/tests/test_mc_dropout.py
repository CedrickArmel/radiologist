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
stateless mc_dropout_predict helper (issue #94).

Tests drive through the public API only. Fixtures build real ONNX models so
no mocks are needed for local code.
"""

import numpy as np
import pytest

CLASSES = ["NORMAL", "ABNORMAL"]


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
