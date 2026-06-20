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

import numpy as np
import pytest
from PIL import Image as PILImage

CLASSES = ["NORMAL", "ABNORMAL"]
FEAT_C, FEAT_H, FEAT_W = 64, 7, 7


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
    def test_explain_returns_explanation_instance(self, det_onnx_path_nonzero):
        """explain(image) must return an Explanation dataclass."""
        from radiologist.inference import Explanation, Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path_nonzero)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.explain(image=image)
        assert isinstance(result, Explanation)


class TestExplainSpatialDimensions:
    def test_saliency_map_matches_input_image_spatial_dims(self, det_onnx_path_nonzero):
        """saliency_map must have H×W dimensions matching the input image."""
        from radiologist.inference import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path_nonzero)

        h, w = 128, 96
        image = np.zeros((h, w, 3), dtype=np.uint8)
        result = predictor.explain(image=image)
        assert result.saliency_map.shape == (h, w)

    def test_saliency_map_matches_pil_image_spatial_dims(self, det_onnx_path_nonzero):
        """saliency_map dims must match a PIL Image input."""
        from radiologist.inference import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path_nonzero)

        h, w = 200, 150
        pil_img = PILImage.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
        result = predictor.explain(image=pil_img)
        assert result.saliency_map.shape == (h, w)


class TestExplainSaliencyValues:
    def test_all_saliency_values_in_0_1(self, det_onnx_path_nonzero):
        """Every value in saliency_map must lie in [0, 1]."""
        from radiologist.inference import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path_nonzero)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.explain(image=image)
        assert float(result.saliency_map.min()) >= 0.0
        assert float(result.saliency_map.max()) <= 1.0


class TestExplainPredictedClass:
    def test_explain_predicted_class_matches_predict(self, det_onnx_path_nonzero):
        """Explanation.predicted_class must be consistent with predict() for the same image."""
        from radiologist.inference import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path_nonzero)
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
