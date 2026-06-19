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

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CLASSES = ["NORMAL", "ABNORMAL"]
INPUT_SHAPE = [1, 3, 224, 224]
N_CLASSES = len(CLASSES)
N_FEATURES = 3 * 224 * 224


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


def build_det_onnx(
    tmp_path,
    priors: Optional[dict] = None,
    filename: str = "model_det.onnx",
) -> str:
    """Build a minimal deterministic 2-class ONNX classifier for tests.

    Outputs: logits (1, N_CLASSES) via Softmax, feature_maps (1, 64, 7, 7) constant.
    Embeds required metadata keys. Optional training_prior embedded when priors given.
    """
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

    extra = {}
    if priors is not None:
        extra["training_prior"] = json.dumps(priors)
    _add_metadata(model, extra)

    path = str(tmp_path / filename)
    onnx.save(model, path)
    return path


def build_mcd_onnx(tmp_path, filename: str = "model_mcd.onnx") -> str:
    """Build a stochastic MCD ONNX model whose logits vary each forward pass.

    Uses RandomUniform so each session.run() produces different logits.
    """
    FLOAT = onnx.TensorProto.FLOAT

    logits_vi = oh.make_tensor_value_info("logits", FLOAT, [1, N_CLASSES])

    rand_node = oh.make_node(
        "RandomUniform",
        inputs=[],
        outputs=["rand_out"],
        dtype=1,
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


@pytest.fixture()
def det_onnx_path(tmp_path):
    return build_det_onnx(tmp_path)


@pytest.fixture()
def mcd_onnx_path(tmp_path):
    return build_mcd_onnx(tmp_path)
