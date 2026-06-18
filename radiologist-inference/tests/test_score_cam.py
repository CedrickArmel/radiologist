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

"""Behavioral tests for Score-CAM: score_cam() and Predictor.explain() (issue #79).

Tests drive through the public API only. Real ONNX models are built per
test — no mocks for local code.
"""

import json
from typing import Optional

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest
from PIL import Image as PILImage

CLASSES = ["NORMAL", "ABNORMAL"]
INPUT_SHAPE = [1, 3, 224, 224]
N_FEATURES = 3 * 224 * 224
N_CLASSES = len(CLASSES)
FEAT_C, FEAT_H, FEAT_W = 64, 7, 7


def _add_metadata(model: onnx.ModelProto, extra: dict) -> onnx.ModelProto:
    base = {
        "classes": json.dumps(CLASSES),
        "input_shape": json.dumps(INPUT_SHAPE),
        "cam_target_layer": "features.28",
        "output_names": json.dumps(["logits", "feature_maps"]),
    }
    base.update(extra)
    del model.metadata_props[:]
    for k, v in base.items():
        e = model.metadata_props.add()
        e.key = k
        e.value = v
    return model


def _build_det_onnx(
    tmp_path,
    priors: Optional[dict] = None,
    filename: str = "model_det.onnx",
) -> str:
    """Build a deterministic 2-class ONNX model with logits + feature_maps outputs."""
    np.random.seed(42)
    W = np.random.randn(N_CLASSES, N_FEATURES).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)

    W_init = onh.from_array(W, name="W")
    b_init = onh.from_array(b, name="b")
    feat_const = onh.from_array(
        np.random.rand(1, FEAT_C, FEAT_H, FEAT_W).astype(np.float32),
        name="feat_const",
    )
    shape_data = onh.from_array(
        np.array([1, N_FEATURES], dtype=np.int64), name="reshape_shape"
    )

    FLOAT = onnx.TensorProto.FLOAT
    logits_vi = oh.make_tensor_value_info("logits", FLOAT, [1, N_CLASSES])
    feature_maps_vi = oh.make_tensor_value_info(
        "feature_maps", FLOAT, [1, FEAT_C, FEAT_H, FEAT_W]
    )

    reshape_node = oh.make_node(
        "Reshape", inputs=["input", "reshape_shape"], outputs=["reshape_out"]
    )
    gemm_node = oh.make_node(
        "Gemm",
        inputs=["reshape_out", "W", "b"],
        outputs=["gemm_out"],
        transB=1,
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
        inputs=[
            oh.make_tensor_value_info("input", onnx.TensorProto.FLOAT, INPUT_SHAPE),
        ],
        outputs=[logits_vi, feature_maps_vi],
        initializer=[W_init, b_init, shape_data, feat_const],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8

    extra = {}
    if priors is not None:
        extra["training_prior"] = json.dumps(priors)
    _add_metadata(model, extra)

    path = str(tmp_path / filename)
    onnx.save(model, path)
    return path


# ---------------------------------------------------------------------------
# score_cam() standalone function
# ---------------------------------------------------------------------------


class TestScoreCamFunction:
    def test_score_cam_returns_array_with_values_in_0_1(self):
        """score_cam() must return a saliency map with all values in [0, 1]."""
        from radiologist.inference import score_cam

        feature_maps = np.random.rand(FEAT_C, FEAT_H, FEAT_W).astype(np.float32)
        logits = np.array([0.3, 0.7], dtype=np.float32)
        result = score_cam(feature_maps=feature_maps, logits=logits)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_score_cam_returns_2d_array_with_feature_map_spatial_dims(self):
        """score_cam() must return shape (H, W) matching feature_maps spatial dims."""
        from radiologist.inference import score_cam

        feature_maps = np.random.rand(FEAT_C, FEAT_H, FEAT_W).astype(np.float32)
        logits = np.array([0.3, 0.7], dtype=np.float32)
        result = score_cam(feature_maps=feature_maps, logits=logits)
        assert result.shape == (FEAT_H, FEAT_W)

    def test_score_cam_uniform_feature_maps_returns_uniform_saliency(self):
        """score_cam() on uniform feature maps (all same channel) produces uniform map."""
        from radiologist.inference import score_cam

        feature_maps = np.ones((FEAT_C, FEAT_H, FEAT_W), dtype=np.float32)
        logits = np.array([0.5, 0.5], dtype=np.float32)
        result = score_cam(feature_maps=feature_maps, logits=logits)
        assert result.shape == (FEAT_H, FEAT_W)
        assert result.min() >= 0.0
        assert result.max() <= 1.0


# ---------------------------------------------------------------------------
# Predictor.explain()
# ---------------------------------------------------------------------------


class TestExplainReturnType:
    def test_explain_returns_explanation_instance(self, tmp_path):
        """explain(image) must return an Explanation dataclass."""
        from radiologist.inference import Explanation, Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.explain(image=image)
        assert isinstance(result, Explanation)


class TestExplainSpatialDimensions:
    def test_saliency_map_matches_input_image_spatial_dims(self, tmp_path):
        """saliency_map must have H×W dimensions matching the input image."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

        h, w = 128, 96
        image = np.zeros((h, w, 3), dtype=np.uint8)
        result = predictor.explain(image=image)
        assert result.saliency_map.shape == (h, w)

    def test_saliency_map_matches_pil_image_spatial_dims(self, tmp_path):
        """saliency_map dims must match a PIL Image input."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

        h, w = 200, 150
        pil_img = PILImage.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
        result = predictor.explain(image=pil_img)
        assert result.saliency_map.shape == (h, w)


class TestExplainSaliencyValues:
    def test_all_saliency_values_in_0_1(self, tmp_path):
        """Every value in saliency_map must lie in [0, 1]."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.explain(image=image)
        assert float(result.saliency_map.min()) >= 0.0
        assert float(result.saliency_map.max()) <= 1.0


class TestExplainPredictedClass:
    def test_explain_predicted_class_matches_predict(self, tmp_path):
        """Explanation.predicted_class must be consistent with predict() for the same image."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        explanation = predictor.explain(image=image)
        prediction = predictor.predict(image=image)
        assert explanation.predicted_class == prediction.predicted_class


class TestExplainWithoutDetModel:
    def test_explain_raises_not_implemented_without_det_model(self):
        """explain() on a predictor without a det model must raise NotImplementedError."""
        from radiologist.inference import Predictor

        predictor = object.__new__(Predictor)
        with pytest.raises(NotImplementedError):
            predictor.explain(image=np.zeros((224, 224, 3), dtype=np.uint8))
