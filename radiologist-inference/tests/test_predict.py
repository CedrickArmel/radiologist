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

"""Behavioral tests for Predictor.from_path and Predictor.predict (issue #77).

Tests drive through the public API only: Predictor.from_path / Predictor.predict.
Fixtures build real ONNX models so no mocks are needed for local code.
"""

import numpy as np
import pytest
from _helpers import build_det_onnx
from PIL import Image as PILImage

CLASSES = ["NORMAL", "ABNORMAL"]


# ---------------------------------------------------------------------------
# RED — failing tests (Step 3)
# ---------------------------------------------------------------------------


class TestPredictorFromPath:
    def test_from_path_returns_predictor_instance(self, det_onnx_path):
        """from_path with a valid ONNX file must return a Predictor instance."""
        from radiologist.inference.predictor import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path)
        assert isinstance(predictor, Predictor)

    def test_from_path_raises_on_unreadable_path(self, tmp_path):
        """from_path with a non-existent file must raise an error."""
        from radiologist.inference.predictor import Predictor

        with pytest.raises((FileNotFoundError, RuntimeError, OSError, Exception)):
            Predictor.from_path(det_path=str(tmp_path / "nonexistent.onnx"))

    def test_from_path_raises_on_invalid_onnx_file(self, tmp_path):
        """from_path with a file that is not a valid ONNX model must raise."""
        from radiologist.inference.predictor import Predictor

        bad_path = tmp_path / "bad.onnx"
        bad_path.write_bytes(b"this is not an onnx model")
        with pytest.raises(Exception):
            Predictor.from_path(det_path=str(bad_path))


class TestPredictReturnType:
    def test_predict_with_file_path_returns_prediction(self, det_onnx_path, tmp_path):
        """predict(image=file_path) must return a Prediction dataclass."""
        from radiologist.inference.predictor import Prediction, Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path)

        img = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB")
        img_path = str(tmp_path / "test_image.png")
        img.save(img_path)

        result = predictor.predict(image=img_path)
        assert isinstance(result, Prediction)

    def test_predict_with_numpy_array_returns_prediction(self, det_onnx_path):
        """predict(image=np.ndarray) must return a Prediction dataclass."""
        from radiologist.inference.predictor import Prediction, Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        assert isinstance(result, Prediction)

    def test_predict_with_pil_image_returns_prediction(self, det_onnx_path):
        """predict(image=PIL.Image) must return a Prediction dataclass."""
        from radiologist.inference.predictor import Prediction, Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path)

        pil_img = PILImage.fromarray(
            np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB"
        )
        result = predictor.predict(image=pil_img)
        assert isinstance(result, Prediction)


class TestPredictProbabilities:
    def test_probabilities_sum_to_one(self, det_onnx_path):
        """Probabilities in Prediction must sum to 1.0 within floating tolerance."""
        from radiologist.inference.predictor import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_probabilities_keyed_by_class_names_from_model_metadata(
        self, det_onnx_path
    ):
        """Prediction.probabilities keys must match the class names in ONNX metadata."""
        from radiologist.inference.predictor import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        assert set(result.probabilities.keys()) == set(CLASSES)

    def test_predicted_class_is_argmax_of_probabilities(self, det_onnx_path):
        """predicted_class must be the class with the highest probability."""
        from radiologist.inference.predictor import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        expected = max(result.probabilities, key=result.probabilities.__getitem__)
        assert result.predicted_class == expected


class TestBayesianPriorCorrection:
    def test_deployment_prior_changes_probabilities(self, det_onnx_path):
        """Supplying deployment_prior must produce different probabilities."""
        from radiologist.inference.predictor import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        no_prior = predictor.predict(image=image)
        with_prior = predictor.predict(
            image=image,
            deployment_prior={"NORMAL": 0.9, "ABNORMAL": 0.1},
        )
        assert no_prior.probabilities != with_prior.probabilities

    def test_deployment_prior_corrected_probs_still_sum_to_one(self, det_onnx_path):
        """Prior-corrected probabilities must still sum to 1.0."""
        from radiologist.inference.predictor import Predictor

        predictor = Predictor.from_path(det_path=det_onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(
            image=image,
            deployment_prior={"NORMAL": 0.9, "ABNORMAL": 0.1},
        )
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_model_embedded_training_prior_applied_when_no_deployment_prior(
        self, tmp_path
    ):
        """When model has training_prior metadata, predict() applies correction."""
        from radiologist.inference.predictor import Predictor

        onnx_path_with_prior = build_det_onnx(
            tmp_path,
            priors={"NORMAL": 0.7, "ABNORMAL": 0.3},
            filename="model_with_prior.onnx",
        )
        onnx_path_no_prior = build_det_onnx(
            tmp_path, priors=None, filename="model_no_prior.onnx"
        )

        pred_with_embedded = Predictor.from_path(det_path=onnx_path_with_prior).predict(
            image=np.zeros((224, 224, 3), dtype=np.uint8)
        )
        pred_no_embedded = Predictor.from_path(det_path=onnx_path_no_prior).predict(
            image=np.zeros((224, 224, 3), dtype=np.uint8)
        )
        assert pred_with_embedded.probabilities != pred_no_embedded.probabilities

    def test_no_prior_returns_raw_softmax_when_model_has_no_embedded_prior(
        self, tmp_path
    ):
        """When model has no training_prior and no deployment_prior is given,
        probabilities come directly from the model's softmax output."""
        from radiologist.inference.predictor import Predictor

        onnx_path = build_det_onnx(tmp_path, priors=None)
        predictor = Predictor.from_path(det_path=onnx_path)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5
        assert set(result.probabilities.keys()) == set(CLASSES)
