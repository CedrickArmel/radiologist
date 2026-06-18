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

"""Behavioral tests for mc_dropout_predict and Predictor.predict_with_uncertainty (issue #80).

Tests drive through the public API only. Fixtures build real ONNX models so
no mocks are needed for local code.
"""

import json
from typing import Optional

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest

CLASSES = ["NORMAL", "ABNORMAL"]
INPUT_SHAPE = [1, 3, 224, 224]
N_CLASSES = len(CLASSES)


def _add_metadata(
    model: onnx.ModelProto, extra: Optional[dict] = None
) -> onnx.ModelProto:
    base = {
        "classes": json.dumps(CLASSES),
        "input_shape": json.dumps(INPUT_SHAPE),
        "cam_target_layer": "features.28",
        "output_names": json.dumps(["logits", "feature_maps"]),
    }
    if extra:
        base.update(extra)
    del model.metadata_props[:]
    for k, v in base.items():
        e = model.metadata_props.add()
        e.key = k
        e.value = v
    return model


def _build_det_onnx(tmp_path, filename: str = "model_det.onnx") -> str:
    """Build a minimal deterministic ONNX model for loading a Predictor."""
    N_FEATURES = 3 * 224 * 224
    np.random.seed(0)
    W = np.random.randn(N_CLASSES, N_FEATURES).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)

    W_init = onh.from_array(W, name="W")
    b_init = onh.from_array(b, name="b")
    feat_const = onh.from_array(
        np.zeros((1, 64, 7, 7), dtype=np.float32), name="feat_const"
    )
    shape_data = onh.from_array(
        np.array([1, N_FEATURES], dtype=np.int64), name="reshape_shape"
    )

    FLOAT = onnx.TensorProto.FLOAT
    logits_vi = oh.make_tensor_value_info("logits", FLOAT, [1, N_CLASSES])
    feature_maps_vi = oh.make_tensor_value_info("feature_maps", FLOAT, [1, 64, 7, 7])

    reshape_node = oh.make_node(
        "Reshape", inputs=["input", "reshape_shape"], outputs=["reshape_out"]
    )
    gemm_node = oh.make_node(
        "Gemm", inputs=["reshape_out", "W", "b"], outputs=["gemm_out"], transB=1
    )
    softmax_node = oh.make_node(
        "Softmax", inputs=["gemm_out"], outputs=["logits"], axis=1
    )
    identity_node = oh.make_node(
        "Identity", inputs=["feat_const"], outputs=["feature_maps"]
    )

    graph = oh.make_graph(
        nodes=[reshape_node, gemm_node, softmax_node, identity_node],
        name="det_classifier",
        inputs=[oh.make_tensor_value_info("input", FLOAT, INPUT_SHAPE)],
        outputs=[logits_vi, feature_maps_vi],
        initializer=[W_init, b_init, shape_data, feat_const],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8
    _add_metadata(model)
    path = str(tmp_path / filename)
    onnx.save(model, path)
    return path


def _build_mcd_onnx(tmp_path, filename: str = "model_mcd.onnx") -> str:
    """Build a stochastic MCD ONNX model whose logits vary each forward pass.

    Architecture: RandomUniform(shape=[1, N_CLASSES]) -> Softmax -> logits
    Each call to session.run() returns different logits, simulating dropout randomness.
    """
    FLOAT = onnx.TensorProto.FLOAT

    logits_vi = oh.make_tensor_value_info("logits", FLOAT, [1, N_CLASSES])

    # RandomUniform produces different values each session.run() call
    rand_node = oh.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["rand_out"],
        dtype=1,  # FLOAT
        shape=[1, N_CLASSES],
    )
    softmax_node = oh.make_node(
        "Softmax", inputs=["rand_out"], outputs=["logits"], axis=1
    )

    graph = oh.make_graph(
        nodes=[rand_node, softmax_node],
        name="mcd_classifier",
        inputs=[oh.make_tensor_value_info("input", FLOAT, INPUT_SHAPE)],
        outputs=[logits_vi],
        initializer=[],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8
    _add_metadata(model)
    path = str(tmp_path / filename)
    onnx.save(model, path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def det_onnx_path(tmp_path):
    return _build_det_onnx(tmp_path)


@pytest.fixture
def mcd_onnx_path(tmp_path):
    return _build_mcd_onnx(tmp_path)


@pytest.fixture
def predictor_with_mcd(tmp_path):
    from radiologist.inference import Predictor

    det = _build_det_onnx(tmp_path)
    mcd = _build_mcd_onnx(tmp_path)
    return Predictor.from_path(det_path=det, mcd_path=mcd)


@pytest.fixture
def predictor_without_mcd(tmp_path):
    from radiologist.inference import Predictor

    det = _build_det_onnx(tmp_path)
    return Predictor.from_path(det_path=det)


@pytest.fixture
def sample_image():
    return np.zeros((224, 224, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Tests: Predictor.predict_with_uncertainty
# ---------------------------------------------------------------------------


class TestPredictWithUncertainty:
    def test_returns_uncertainty_result_type(self, predictor_with_mcd, sample_image):
        """predict_with_uncertainty must return an UncertaintyResult."""
        from radiologist.inference import UncertaintyResult

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
        with pytest.raises(RuntimeError, match="mcd_path"):
            predictor_without_mcd.predict_with_uncertainty(sample_image)


# ---------------------------------------------------------------------------
# Tests: mc_dropout_predict (public API function)
# ---------------------------------------------------------------------------


class TestMcDropoutPredict:
    def test_returns_uncertainty_result(self, mcd_onnx_path, sample_image):
        """mc_dropout_predict must return an UncertaintyResult."""
        import onnxruntime as ort

        from radiologist.inference import UncertaintyResult, mc_dropout_predict

        session = ort.InferenceSession(mcd_onnx_path)
        preprocessed = sample_image.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        result = mc_dropout_predict(session, preprocessed, n_passes=10)
        assert isinstance(result, UncertaintyResult)

    def test_mean_probs_sum_to_one(self, mcd_onnx_path, sample_image):
        """mc_dropout_predict mean_probabilities must sum to 1.0."""
        import onnxruntime as ort

        from radiologist.inference import mc_dropout_predict

        session = ort.InferenceSession(mcd_onnx_path)
        preprocessed = sample_image.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        result = mc_dropout_predict(session, preprocessed, n_passes=10)
        total = sum(result.mean_probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_n_passes_recorded_in_result(self, mcd_onnx_path, sample_image):
        """mc_dropout_predict result must record the requested number of passes."""
        import onnxruntime as ort

        from radiologist.inference import mc_dropout_predict

        session = ort.InferenceSession(mcd_onnx_path)
        preprocessed = sample_image.astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        result = mc_dropout_predict(session, preprocessed, n_passes=20)
        assert result.n_passes == 20
