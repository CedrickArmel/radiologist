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

from typing import Literal

import torch.nn as nn


def initialize_weights(
    module: nn.Module,
    dist: str,
    a: float = 0.1,
    mode: Literal["fan_in", "fan_out"] = "fan_out",
    nonlinearity: Literal[
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "sigmoid",
        "tanh",
        "relu",
        "leaky_relu",
        "selu",
    ] = "leaky_relu",
) -> None:
    """Xavier-init Linear layers and kaiming-init Conv2d/Conv3d layers.

    Args:
        module: The network module whose submodules will be initialised.
        dist: Distribution to use — "normal" or "uniform".
        a: Negative slope for leaky_relu (kaiming only).
        mode: Fan mode for kaiming init.
        nonlinearity: Nonlinearity name for kaiming init.

    Raises:
        ValueError: If dist is not "normal" or "uniform".
    """
    if dist not in ("normal", "uniform"):
        raise ValueError(f"dist must be 'normal' or 'uniform', got '{dist}'")

    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if dist == "normal":
                nn.init.kaiming_normal_(
                    m.weight, a=a, mode=mode, nonlinearity=nonlinearity
                )
            else:
                nn.init.kaiming_uniform_(
                    m.weight, a=a, mode=mode, nonlinearity=nonlinearity
                )
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            if dist == "normal":
                nn.init.xavier_normal_(m.weight)
            else:
                nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
