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

"""Tests for the radiologist.inference CLI.

All tests use typer.testing.CliRunner and drive real Predictor and
WandbRegistry instances. Only the W&B SDK boundary (_wandb sentinel) is
mocked, and no radiologist.* class is mocked.
"""

import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest
from PIL import Image as PILImage  # type: ignore[import-untyped]

from radiologist.inference.cli import app

CLASSES = ["NORMAL", "ABNORMAL"]
INPUT_SHAPE = [1, 3, 224, 224]
N_CLASSES = len(CLASSES)
N_FEATURES = 3 * 224 * 224


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
        inputs=[oh.make_tensor_value_info("input", FLOAT, INPUT_SHAPE)],
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
    mock_wandb = MagicMock()
    artifact = MagicMock()
    artifact.download.return_value = os.path.dirname(onnx_path)
    api_instance = MagicMock()
    api_instance.artifact.return_value = artifact
    mock_wandb.Api.return_value = api_instance
    return mock_wandb


def _save_png(tmp_path, filename="chest.png"):
    img_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    img_path = str(tmp_path / filename)
    PILImage.fromarray(img_arr).save(img_path)
    return img_path


def _runner():
    from typer.testing import CliRunner

    return CliRunner()


def test_predict_exits_0_on_valid_image_and_model(tmp_path):
    """predict command exits 0 and prints the class when given a real model and image."""
    det_path = _build_det_onnx(tmp_path)
    img_path = _save_png(tmp_path)

    result = _runner().invoke(
        app,
        ["predict", img_path, "--model", det_path],
    )

    assert result.exit_code == 0
    assert "Predicted class:" in result.output


def test_predict_exits_1_when_model_path_does_not_exist(tmp_path):
    """predict command exits 1 when the model file does not exist."""
    img_path = _save_png(tmp_path)

    result = _runner().invoke(
        app,
        ["predict", img_path, "--model", str(tmp_path / "nonexistent.onnx")],
    )

    assert result.exit_code == 1


def test_predict_exits_1_when_image_path_does_not_exist(tmp_path):
    """predict command exits 1 when the image file does not exist."""
    det_path = _build_det_onnx(tmp_path)

    result = _runner().invoke(
        app,
        [
            "predict",
            str(tmp_path / "nonexistent_image.jpg"),
            "--model",
            det_path,
        ],
    )

    assert result.exit_code == 1


def test_pull_exits_0_on_valid_artifact(tmp_path):
    """pull command exits 0 when artifact is retrievable via real WandbRegistry."""
    import radiologist.registry.resolver as resolver_mod

    onnx_path = _build_det_onnx(tmp_path)
    mock_wandb = _make_wandb_mock(onnx_path)

    with patch.object(resolver_mod, "_wandb", mock_wandb):
        result = _runner().invoke(
            app,
            ["pull", "entity/project/name:v1", "--local-dir", str(tmp_path)],
        )

    assert result.exit_code == 0
    assert "Model downloaded to:" in result.output


def test_pull_exits_1_when_wandb_sdk_is_absent(tmp_path):
    """pull command exits 1 when the W&B SDK is absent (real registry, _wandb=None)."""
    import radiologist.registry.optional as optional_mod

    with patch.object(optional_mod, "_wandb", None):
        result = _runner().invoke(
            app,
            ["pull", "entity/project/name:v0", "--local-dir", str(tmp_path)],
        )

    assert result.exit_code == 1


def test_cli_entry_point_raises_runtime_error_when_typer_absent():
    """Invoking cli entry point without typer raises RuntimeError naming 'cli' extra."""
    import radiologist.inference.cli as cli_mod

    with patch.object(cli_mod, "_typer", None):
        with pytest.raises(RuntimeError, match="cli"):
            cli_mod.main()
