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

from __future__ import annotations

import json
import os
from typing import List, Tuple

import torch
import torch.nn as nn

try:
    import onnx  # type: ignore[import-untyped]
except ImportError:
    onnx = None  # type: ignore[assignment]

from radiologist.core.module import LModule
from radiologist.core.registry.promote import (
    _CamWrapper,
    _resolve_layer,
    _set_metadata_props,
)
from radiologist.registry import ExportResult


def export_onnx(
    ckpt_path: str,
    run_id: str,
    input_shape: Tuple[int, ...],
    classes: List[str],
    cam_target_layer: str,
    out_dir: str,
    opset: int = 18,
) -> ExportResult:
    """Load a Lightning checkpoint and write deterministic + MC-Dropout ONNX files.

    Returns an ExportResult with both local ONNX paths plus run_id, input_shape,
    and classes (no W&B interaction).
    """
    if onnx is None:
        raise RuntimeError(
            "onnx is required for export_onnx. " "Install it with: pip install onnx"
        )

    os.makedirs(out_dir, exist_ok=True)

    lmodule = LModule.load_from_checkpoint(
        ckpt_path, map_location="cpu", weights_only=False
    )
    net = lmodule.net

    target_layer = _resolve_layer(net, cam_target_layer)

    base_meta = {
        "run_id": run_id,
        "input_shape": json.dumps(list(input_shape)),
        "classes": json.dumps(classes),
        "framework": "pytorch-lightning",
    }

    dummy_input = torch.zeros(*input_shape)

    # --- Export 1: deterministic (dropout folded, cam hook active) ----------
    det_path = os.path.join(out_dir, f"model-{run_id}.onnx")

    net.train(mode=False)
    wrapper = _CamWrapper(net, target_layer)
    wrapper.train(mode=False)

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        det_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["logits", "feature_maps"],
        do_constant_folding=True,
    )

    det_model = onnx.load(det_path)
    det_meta = dict(base_meta)
    det_meta["cam_target_layer"] = cam_target_layer
    det_meta["output_names"] = json.dumps(["logits", "feature_maps"])
    _set_metadata_props(det_model, det_meta)
    onnx.save(det_model, det_path)

    # --- Export 2: MC-Dropout (Dropout preserved in training mode) ----------
    mcd_path = os.path.join(out_dir, f"model-{run_id}-mcd.onnx")

    net.train(mode=False)
    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    torch.onnx.export(
        net,
        (dummy_input,),
        mcd_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["logits"],
        training=torch.onnx.TrainingMode.PRESERVE,
        do_constant_folding=False,
    )

    mcd_model = onnx.load(mcd_path)
    mcd_meta = dict(base_meta)
    mcd_meta["mc_dropout"] = "true"
    _set_metadata_props(mcd_model, mcd_meta)
    onnx.save(mcd_model, mcd_path)

    return ExportResult(
        det_path=det_path,
        mcd_path=mcd_path,
        run_id=run_id,
        input_shape=input_shape,
        classes=classes,
    )
