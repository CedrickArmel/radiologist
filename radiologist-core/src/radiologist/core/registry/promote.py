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

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    import onnx  # type: ignore[import-untyped]
except ImportError:
    onnx = None  # type: ignore[assignment]


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
