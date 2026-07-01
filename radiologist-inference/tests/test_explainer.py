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

"""Behavioral tests for Explainer (issue #93).

Tests drive through the public API only: Explainer.from_path, Explainer.explain,
and the inherited Explainer.predict. Real ONNX models are built per test — no
mocks for locally owned code.
"""

import numpy as np
from PIL import Image as PILImage

from radiologist.inference import Classifier, Explainer, Explanation, Prediction

CLASSES = ["NORMAL", "ABNORMAL"]


class TestExplainerFromPath:
    def test_from_path_returns_explainer_instance(self, det_onnx_path_nonzero):
        """from_path called on Explainer must return an Explainer instance."""
        explainer = Explainer.from_path(det_path=det_onnx_path_nonzero)
        assert isinstance(explainer, Explainer)
        assert isinstance(explainer, Classifier)


class TestExplainSpatialDimensions:
    def test_saliency_map_matches_input_image_spatial_dims(self, det_onnx_path_nonzero):
        """saliency_map must have H x W dimensions matching the original image."""
        explainer = Explainer.from_path(det_path=det_onnx_path_nonzero)

        h, w = 128, 96
        image = np.zeros((h, w, 3), dtype=np.uint8)
        result = explainer.explain(image=image)
        assert isinstance(result, Explanation)
        assert result.saliency_map.shape == (h, w)

    def test_saliency_map_matches_pil_image_spatial_dims(self, det_onnx_path_nonzero):
        """saliency_map dims must match a PIL Image input, not the model input shape."""
        explainer = Explainer.from_path(det_path=det_onnx_path_nonzero)

        h, w = 200, 150
        pil_img = PILImage.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
        result = explainer.explain(image=pil_img)
        assert result.saliency_map.shape == (h, w)


class TestExplainPredictedClass:
    def test_explain_predicted_class_matches_predict(self, det_onnx_path_nonzero):
        """Explanation.predicted_class must agree with predict() for the same image."""
        explainer = Explainer.from_path(det_path=det_onnx_path_nonzero)
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        explanation = explainer.explain(image=image)
        prediction = explainer.predict(image=image)

        assert explanation.predicted_class in CLASSES
        assert explanation.predicted_class == prediction.predicted_class


class TestExplainSaliencyValues:
    def test_all_saliency_values_in_0_1(self, det_onnx_path_nonzero):
        """Every value in saliency_map must lie in [0, 1]."""
        explainer = Explainer.from_path(det_path=det_onnx_path_nonzero)
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        result = explainer.explain(image=image)

        assert float(result.saliency_map.min()) >= 0.0
        assert float(result.saliency_map.max()) <= 1.0


class TestExplainerInheritsPredict:
    def test_explainer_also_serves_predict(self, det_onnx_path_nonzero):
        """An Explainer instance must also answer predict() and return a Prediction."""
        explainer = Explainer.from_path(det_path=det_onnx_path_nonzero)
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        result = explainer.predict(image=image)

        assert isinstance(result, Prediction)
        assert set(result.probabilities.keys()) == set(CLASSES)
