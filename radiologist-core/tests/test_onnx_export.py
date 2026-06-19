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
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from radiologist.core import LModule

import pytest
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassFBetaScore

# ---------------------------------------------------------------------------
# Helpers — minimal net and LModule (same as test_registry.py)
# ---------------------------------------------------------------------------


def _make_tiny_net() -> nn.Sequential:
    """Tiny net with a Dropout so MCD export can find it."""
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 2),
    )


def _make_lmodule(net: nn.Module) -> "LModule":
    from radiologist.core import FocalLoss, LModule

    return LModule(
        net=net,
        loss=FocalLoss(),
        metric=partial(MulticlassFBetaScore, beta=1.0, num_classes=2),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
    )


# ---------------------------------------------------------------------------
# Fixture: run export_onnx with a real tiny model
# ---------------------------------------------------------------------------

INPUT_SHAPE = (1, 3, 8, 8)
CLASSES = ["healthy", "sick"]
RUN_ID = "testrun001"
CAM_TARGET_LAYER = "2"  # Dropout layer in the sequential net


@pytest.fixture()
def export_result(tmp_path: Path):
    """Run export_onnx against a real tiny LModule; return (result, tmp_path)."""
    import radiologist.core.registry.export as export_mod

    net = _make_tiny_net()
    lm = _make_lmodule(net)

    with patch.object(export_mod.LModule, "load_from_checkpoint", return_value=lm):
        result = export_mod.export_onnx(
            ckpt_path="fake_ckpt.ckpt",
            run_id=RUN_ID,
            input_shape=INPUT_SHAPE,
            classes=CLASSES,
            cam_target_layer=CAM_TARGET_LAYER,
            out_dir=str(tmp_path),
        )
    return result, tmp_path


# ---------------------------------------------------------------------------
# AC1: writes two ONNX files and returns ExportResult pointing at them
# ---------------------------------------------------------------------------


def test_export_onnx_writes_two_onnx_files_and_returns_paths(export_result):
    result, tmp_path = export_result
    assert Path(result.det_path).exists(), "det_path must exist on disk"
    assert Path(result.mcd_path).exists(), "mcd_path must exist on disk"
    assert result.det_path.endswith(".onnx")
    assert result.mcd_path.endswith(".onnx")
    assert result.det_path != result.mcd_path


# ---------------------------------------------------------------------------
# AC2: ExportResult carries the passed-in run_id, input_shape, classes
# ---------------------------------------------------------------------------


def test_export_onnx_result_carries_caller_metadata(export_result):
    result, _ = export_result
    assert result.run_id == RUN_ID
    assert result.input_shape == INPUT_SHAPE
    assert result.classes == CLASSES


# ---------------------------------------------------------------------------
# AC3: deterministic ONNX has required output names and metadata props
# ---------------------------------------------------------------------------


def test_deterministic_onnx_has_logits_feature_maps_outputs_and_metadata(export_result):
    import onnx

    result, _ = export_result
    model = onnx.load(result.det_path)
    output_names = [o.name for o in model.graph.output]
    assert "logits" in output_names
    assert "feature_maps" in output_names

    props = {p.key: p.value for p in model.metadata_props}
    for key in ("run_id", "input_shape", "classes", "framework"):
        assert key in props, f"Missing metadata key: {key}"
    assert "cam_target_layer" in props
    assert json.loads(props["output_names"]) == ["logits", "feature_maps"]


# ---------------------------------------------------------------------------
# AC4: MC-Dropout ONNX retains Dropout nodes and has mc_dropout metadata
# ---------------------------------------------------------------------------


def test_mcd_onnx_retains_dropout_nodes_and_has_mc_dropout_metadata(export_result):
    import onnx

    result, _ = export_result
    model = onnx.load(result.mcd_path)
    dropout_nodes = [n for n in model.graph.node if n.op_type == "Dropout"]
    assert len(dropout_nodes) > 0, "MCD ONNX must retain at least one Dropout node"

    props = {p.key: p.value for p in model.metadata_props}
    assert props.get("mc_dropout") == "true"


# ---------------------------------------------------------------------------
# AC5: export_onnx performs no W&B calls (import does not require wandb)
# ---------------------------------------------------------------------------


def test_export_onnx_does_not_import_or_call_wandb(tmp_path: Path):
    """Patch wandb to None at module level; export_onnx must still succeed."""
    import radiologist.core.registry.export as export_mod

    net = _make_tiny_net()
    lm = _make_lmodule(net)

    with (
        patch.object(export_mod, "wandb", None, create=True),
        patch.object(export_mod.LModule, "load_from_checkpoint", return_value=lm),
    ):
        result = export_mod.export_onnx(
            ckpt_path="fake_ckpt.ckpt",
            run_id=RUN_ID,
            input_shape=INPUT_SHAPE,
            classes=CLASSES,
            cam_target_layer=CAM_TARGET_LAYER,
            out_dir=str(tmp_path),
        )

    assert Path(result.det_path).exists()
    assert Path(result.mcd_path).exists()


# ---------------------------------------------------------------------------
# AC: export_onnx importable from radiologist.core.registry
# ---------------------------------------------------------------------------


def test_export_onnx_importable_from_core_registry():
    from radiologist.core.registry import export_onnx

    assert callable(export_onnx)


# ---------------------------------------------------------------------------
# AC: AttributeError raised for bad cam_target_layer
# ---------------------------------------------------------------------------


def test_export_onnx_raises_attribute_error_for_bad_cam_layer(tmp_path: Path):
    import radiologist.core.registry.export as export_mod

    net = _make_tiny_net()
    lm = _make_lmodule(net)

    with patch.object(export_mod.LModule, "load_from_checkpoint", return_value=lm):
        with pytest.raises(AttributeError):
            export_mod.export_onnx(
                ckpt_path="fake_ckpt.ckpt",
                run_id=RUN_ID,
                input_shape=INPUT_SHAPE,
                classes=CLASSES,
                cam_target_layer="nonexistent.deep.layer",
                out_dir=str(tmp_path),
            )
