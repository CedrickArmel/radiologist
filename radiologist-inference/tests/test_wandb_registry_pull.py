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

"""Behavioral tests for inference model-pull via WandbRegistry.pull() (issue #90).

All tests drive through the public API only. wandb SDK is mocked at the
process boundary since it requires external network/auth.
"""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest

from radiologist.registry.wandb_registry import WandbRegistry

CLASSES = ["NORMAL", "ABNORMAL"]
INPUT_SHAPE = [1, 3, 224, 224]
N_FEATURES = 3 * 224 * 224
N_CLASSES = len(CLASSES)


def _build_det_onnx(tmp_path, filename="model_det.onnx"):
    np.random.seed(42)
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

    graph = oh.make_graph(
        nodes=[
            oh.make_node(
                "Reshape",
                inputs=["input", "reshape_shape"],
                outputs=["reshape_out"],
            ),
            oh.make_node(
                "Gemm",
                inputs=["reshape_out", "W", "b"],
                outputs=["gemm_out"],
                transB=1,
            ),
            oh.make_node("Softmax", inputs=["gemm_out"], outputs=["logits"], axis=1),
            oh.make_node("Identity", inputs=["feat_const"], outputs=["feature_maps"]),
        ],
        name="det_classifier",
        inputs=[
            oh.make_tensor_value_info("input", FLOAT, INPUT_SHAPE),
        ],
        outputs=[logits_vi, feature_maps_vi],
        initializer=[W_init, b_init, shape_data, feat_const],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model.ir_version = 8

    base_meta = {
        "classes": json.dumps(CLASSES),
        "input_shape": json.dumps(INPUT_SHAPE),
        "cam_target_layer": "features.28",
        "output_names": json.dumps(["logits", "feature_maps"]),
    }
    del model.metadata_props[:]
    for k, v in base_meta.items():
        e = model.metadata_props.add()
        e.key = k
        e.value = v

    path = str(tmp_path / filename)
    onnx.save(model, path)
    return path


def _make_wandb_mock(onnx_path):
    """Return a wandb mock whose Api().artifact().download() places onnx_path."""
    mock_wandb = MagicMock()

    artifact = MagicMock()
    artifact.download.return_value = os.path.dirname(onnx_path)

    api_instance = MagicMock()
    api_instance.artifact.return_value = artifact
    mock_wandb.Api.return_value = api_instance

    return mock_wandb


class TestWandbRegistryPull:
    def test_pull_returns_path_to_onnx_file(self, tmp_path):
        """WandbRegistry.pull() must return the local path to an .onnx file."""
        import radiologist.registry.resolver as resolver_mod

        onnx_path = _build_det_onnx(tmp_path)
        mock_wandb = _make_wandb_mock(onnx_path)

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            reg = WandbRegistry()
            result = reg.pull(
                artifact_path="entity/project/name:v0",
                local_dir=str(tmp_path),
            )

        assert result.endswith(".onnx")

    def test_pull_file_exists_at_returned_path(self, tmp_path):
        """The path returned by WandbRegistry.pull() must point to an existing file."""
        import radiologist.registry.resolver as resolver_mod

        onnx_path = _build_det_onnx(tmp_path)
        mock_wandb = _make_wandb_mock(onnx_path)

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            reg = WandbRegistry()
            result = reg.pull(
                artifact_path="entity/project/name:v0",
                local_dir=str(tmp_path),
            )

        assert os.path.isfile(result)

    def test_pull_raises_runtime_error_when_wandb_absent(self, tmp_path):
        """WandbRegistry.pull() raises RuntimeError when wandb is absent."""
        import radiologist.registry.resolver as resolver_mod

        with patch.object(resolver_mod, "_wandb", None):
            with pytest.raises(RuntimeError):
                reg = WandbRegistry()
                reg.pull(
                    artifact_path="entity/project/name:v0",
                    local_dir=str(tmp_path),
                )


class TestPredictorFromRegistryViaWandbRegistry:
    def test_from_registry_returns_predictor_instance(self, tmp_path):
        """Predictor.from_registry must return a Predictor instance."""
        import radiologist.registry.resolver as resolver_mod
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        mock_wandb = _make_wandb_mock(onnx_path)

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            predictor = Predictor.from_registry(
                artifact_path="entity/project/name:v0",
                local_dir=str(tmp_path),
            )

        assert isinstance(predictor, Predictor)

    def test_from_registry_predictor_produces_prediction(self, tmp_path):
        """Predictor returned by from_registry must produce a Prediction on predict()."""
        import radiologist.registry.resolver as resolver_mod
        from radiologist.inference import Prediction, Predictor

        onnx_path = _build_det_onnx(tmp_path)
        mock_wandb = _make_wandb_mock(onnx_path)

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            predictor = Predictor.from_registry(
                artifact_path="entity/project/name:v0",
                local_dir=str(tmp_path),
            )

        image = np.zeros((224, 224, 3), dtype=np.uint8)
        result = predictor.predict(image=image)
        assert isinstance(result, Prediction)

    def test_from_registry_prediction_shape_matches_from_path(self, tmp_path):
        """from_registry predictor must produce Prediction with same keys as from_path."""
        import radiologist.registry.resolver as resolver_mod
        from radiologist.inference import Predictor

        onnx_path = _build_det_onnx(tmp_path)
        mock_wandb = _make_wandb_mock(onnx_path)

        with patch.object(resolver_mod, "_wandb", mock_wandb):
            registry_predictor = Predictor.from_registry(
                artifact_path="entity/project/name:v0",
                local_dir=str(tmp_path),
            )

        path_predictor = Predictor.from_path(det_path=onnx_path)
        image = np.zeros((224, 224, 3), dtype=np.uint8)

        registry_result = registry_predictor.predict(image=image)
        path_result = path_predictor.predict(image=image)

        assert set(registry_result.probabilities.keys()) == set(
            path_result.probabilities.keys()
        )
        assert registry_result.predicted_class == path_result.predicted_class

    def test_from_registry_raises_runtime_error_when_wandb_absent(self, tmp_path):
        """from_registry raises RuntimeError when wandb absent."""
        import radiologist.registry.resolver as resolver_mod
        from radiologist.inference import Predictor

        with patch.object(resolver_mod, "_wandb", None):
            with pytest.raises(RuntimeError):
                Predictor.from_registry(
                    artifact_path="entity/project/name:v0",
                    local_dir=str(tmp_path),
                )
