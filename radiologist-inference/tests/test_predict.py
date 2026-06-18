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
N_FEATURES = 3 * 224 * 224  # flattened input tensor
N_CLASSES = len(CLASSES)


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
    """Build a deterministic 2-class ONNX model with real softmax output.

    Architecture: flatten -> Gemm(W, b) -> Softmax (logits output)
    Also emits a dummy feature_maps output (zeros) for CAM compatibility.

    Metadata embeds classes, input_shape, cam_target_layer, output_names.
    Optionally embeds training_prior.
    """
    # Weight and bias for Gemm: W shape (N_CLASSES, N_FEATURES), b shape (N_CLASSES,)
    np.random.seed(42)
    W = np.random.randn(N_CLASSES, N_FEATURES).astype(np.float32)
    b = np.zeros(N_CLASSES, dtype=np.float32)

    W_init = onh.from_array(W, name="W")
    b_init = onh.from_array(b, name="b")
    # Dummy constant feature_maps (1, 64, 7, 7) = zeros
    feat_const = onh.from_array(
        np.zeros((1, 64, 7, 7), dtype=np.float32), name="feat_const"
    )

    # Reshape input from (1,3,224,224) -> (1, N_FEATURES)
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
# RED — failing tests (Step 3)
# ---------------------------------------------------------------------------


class TestPredictorFromPath:
    def test_from_path_returns_predictor_instance(self, tmp_path):
        """from_path with a valid ONNX file must return a Predictor instance."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)
        assert isinstance(predictor, Predictor)

    def test_from_path_raises_on_unreadable_path(self, tmp_path):
        """from_path with a non-existent file must raise an error."""
        from radiologist.inference import Predictor

        with pytest.raises((FileNotFoundError, RuntimeError, OSError, Exception)):
            Predictor.from_path(det_path=str(tmp_path / "nonexistent.onnx"))

    def test_from_path_raises_on_invalid_onnx_file(self, tmp_path):
        """from_path with a file that is not a valid ONNX model must raise."""
        from radiologist.inference import Predictor

        bad_path = tmp_path / "bad.onnx"
        bad_path.write_bytes(b"this is not an onnx model")
        with pytest.raises(Exception):
            Predictor.from_path(det_path=str(bad_path))


class TestPredictReturnType:
    def test_predict_with_file_path_returns_prediction(self, tmp_path):
        """predict(image=file_path) must return a Prediction dataclass."""
        from radiologist.inference import Prediction, Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

        img = PILImage.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB")
        img_path = str(tmp_path / "test_image.png")
        img.save(img_path)

        result = predictor.predict(image=img_path)
        assert isinstance(result, Prediction)

    def test_predict_with_numpy_array_returns_prediction(self, tmp_path):
        """predict(image=np.ndarray) must return a Prediction dataclass."""
        from radiologist.inference import Prediction, Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        assert isinstance(result, Prediction)

    def test_predict_with_pil_image_returns_prediction(self, tmp_path):
        """predict(image=PIL.Image) must return a Prediction dataclass."""
        from radiologist.inference import Prediction, Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

        pil_img = PILImage.fromarray(
            np.zeros((224, 224, 3), dtype=np.uint8), mode="RGB"
        )
        result = predictor.predict(image=pil_img)
        assert isinstance(result, Prediction)


class TestPredictProbabilities:
    def test_probabilities_sum_to_one(self, tmp_path):
        """Probabilities in Prediction must sum to 1.0 within floating tolerance."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        total = sum(result.probabilities.values())
        assert abs(total - 1.0) < 1e-5

    def test_probabilities_keyed_by_class_names_from_model_metadata(self, tmp_path):
        """Prediction.probabilities keys must match the class names in ONNX metadata."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        assert set(result.probabilities.keys()) == set(CLASSES)

    def test_predicted_class_is_argmax_of_probabilities(self, tmp_path):
        """predicted_class must be the class with the highest probability."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        expected = max(result.probabilities, key=result.probabilities.__getitem__)
        assert result.predicted_class == expected


class TestBayesianPriorCorrection:
    def test_deployment_prior_changes_probabilities(self, tmp_path):
        """Supplying deployment_prior must produce different probabilities."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        no_prior = predictor.predict(image=image)
        with_prior = predictor.predict(
            image=image,
            deployment_prior={"NORMAL": 0.9, "ABNORMAL": 0.1},
        )
        assert no_prior.probabilities != with_prior.probabilities

    def test_deployment_prior_corrected_probs_still_sum_to_one(self, tmp_path):
        """Prior-corrected probabilities must still sum to 1.0."""
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        predictor = Predictor.from_path(det_path=onnx_path)

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
        from radiologist.inference import Predictor

        onnx_path_with_prior = _build_det_onnx(
            tmp_path,
            priors={"NORMAL": 0.7, "ABNORMAL": 0.3},
            filename="model_with_prior.onnx",
        )
        onnx_path_no_prior = _build_det_onnx(
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
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path, priors=None)
        predictor = Predictor.from_path(det_path=onnx_path)
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        # Probabilities must still sum to 1 and be keyed by class names
        assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5
        assert set(result.probabilities.keys()) == set(CLASSES)
