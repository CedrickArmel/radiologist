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

# MIT License
#
# Copyright (c) 2026 @CedrickArmel, @TaxelleT, @Yeyecodes
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
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import wandb  # type: ignore[import-untyped]
except ImportError:
    wandb = None  # type: ignore[assignment]

try:
    import onnx  # type: ignore[import-untyped]
    import onnx.helper as onnx_helper  # type: ignore[import-untyped]
except ImportError:
    onnx = None  # type: ignore[assignment]
    onnx_helper = None  # type: ignore[assignment]

from radiologist.core.module import LModule
from radiologist.core.registry.pull import pull_checkpoint


def _resolve_layer(net: nn.Module, dot_path: str) -> nn.Module:
    """Resolve a dot-separated attribute path against ``net``.

    Raises:
        AttributeError: if any segment of the path does not exist.
    """
    obj = net
    for part in dot_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise AttributeError(
                f"Layer {dot_path!r} not found in net: "
                f"segment {part!r} missing on {type(obj).__name__}"
            )
    return obj  # type: ignore[return-value]


class _CamWrapper(nn.Module):
    """Thin wrapper that returns ``(logits, activation)`` from a forward pass.

    The activation is captured from ``target_layer`` via a forward hook.
    """

    def __init__(self, net: nn.Module, target_layer: nn.Module) -> None:
        super().__init__()
        self.net = net
        self._activation: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._hook)

    def _hook(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self._activation = output

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        if self._activation is None:
            raise RuntimeError("Forward hook did not capture activation.")
        return logits, self._activation


def _set_metadata_props(model: "onnx.ModelProto", props: dict) -> "onnx.ModelProto":
    """Write ``props`` dict as ``metadata_props`` on an ONNX model."""
    del model.metadata_props[:]
    for k, v in props.items():
        entry = model.metadata_props.add()
        entry.key = k
        entry.value = v
    return model


def promote_to_registry(
    artifact: str,
    collection: str,
    registry_alias: str,
    input_shape: Tuple[int, ...],
    classes: List[str],
    cam_target_layer: str,
    local_dir: str,
    opset: int = 18,
) -> str:
    """Pull a checkpoint, export two ONNX models, and link them to the W&B registry.

    Args:
        artifact: W&B artifact reference ``"entity/project/name:alias"``.
        collection: W&B Model Registry collection name.
        registry_alias: alias to attach when linking (e.g. ``"latest"``).
        input_shape: ONNX input shape, e.g. ``(1, 3, 224, 224)``.
        classes: ordered class names embedded in ONNX metadata.
        cam_target_layer: dot-path into ``LModule.net`` for CAM activation hook.
        local_dir: directory for downloaded checkpoint and exported ONNX files.
        opset: ONNX opset version (default 18, must be ≥ 12 for MCD).

    Returns:
        Qualified name of the linked artifact in the W&B Model Registry.

    Raises:
        RuntimeError: if ``wandb`` or ``onnx`` is not installed.
        AttributeError: if ``cam_target_layer`` cannot be resolved against ``LModule.net``.
    """
    if wandb is None:
        raise RuntimeError(
            "wandb is required for promote_to_registry. "
            "Install it with: pip install wandb"
        )
    if onnx is None:
        raise RuntimeError(
            "onnx is required for promote_to_registry. "
            "Install it with: pip install onnx"
        )

    os.makedirs(local_dir, exist_ok=True)

    # --- Pull checkpoint and read run metadata ---------------------------
    ckpt_path = pull_checkpoint(artifact, local_dir)

    api = wandb.Api()
    art = api.artifact(artifact)
    source_run = art.logged_by()
    run_id: str = source_run.id
    precision: str = source_run.config["trainer"]["precision"]

    # --- Load LModule from checkpoint ------------------------------------
    # Lightning checkpoints embed Python objects (e.g. functools.partial) so we
    # must opt out of the weights_only=True default introduced in PyTorch 2.6.
    # This is safe because the checkpoint originates from our own W&B artifact.
    lmodule = LModule.load_from_checkpoint(
        ckpt_path, map_location="cpu", weights_only=False
    )
    net = lmodule.net

    # --- Resolve cam_target_layer (raises AttributeError if missing) -----
    target_layer = _resolve_layer(net, cam_target_layer)

    # --- Common metadata -------------------------------------------------
    base_meta = {
        "precision": precision,
        "run_id": run_id,
        "input_shape": json.dumps(list(input_shape)),
        "classes": json.dumps(classes),
        "framework": "pytorch-lightning",
    }

    dummy_input = torch.zeros(*input_shape)

    # =====================================================================
    # Export 1 — deterministic (dropout folded, cam hook active)
    # =====================================================================
    det_path = os.path.join(local_dir, f"model-{run_id}.onnx")

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

    # =====================================================================
    # Export 2 — MC-dropout (Dropout preserved in training mode)
    # =====================================================================
    mcd_path = os.path.join(local_dir, f"model-{run_id}-mcd.onnx")

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

    # =====================================================================
    # Upload and link both artifacts to W&B Model Registry
    # =====================================================================
    with wandb.init(job_type="registry-promote") as run:
        det_art = wandb.Artifact(
            name=f"model-{run_id}",
            type="model",
        )
        det_art.add_file(det_path)
        run.log_artifact(det_art)
        det_art.link(collection, aliases=[registry_alias])

        mcd_art = wandb.Artifact(
            name=f"model-{run_id}-mcd",
            type="model",
        )
        mcd_art.add_file(mcd_path)
        linked = run.log_artifact(mcd_art)
        linked.wait()
        mcd_art.link(collection, aliases=[registry_alias])
        qualified_name: str = linked.qualified_name

    return qualified_name
