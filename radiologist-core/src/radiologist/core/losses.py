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

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for multi-class classification with class-index targets.

    Args:
        to_onehot_y: convert integer target to one-hot before loss computation.
        gamma: focusing exponent.
        alpha: overall scaling factor.
        reduction: ``"mean"``, ``"sum"``, or ``"none"``.
        use_softmax: apply softmax to logits before computing probabilities.
    """

    _VALID_REDUCTIONS = {"mean", "sum", "none"}

    def __init__(
        self,
        to_onehot_y: bool = False,
        gamma: float = 2.0,
        alpha: float = 1.0,
        reduction: str = "mean",
        use_softmax: bool = True,
    ) -> None:
        if reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"reduction must be one of {self._VALID_REDUCTIONS}, got {reduction!r}"
            )
        super().__init__()
        self.to_onehot_y = to_onehot_y
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.use_softmax = use_softmax

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: ``(N, C)`` unnormalized class scores.
            target: ``(N,)`` integer class indices or ``(N, C)`` one-hot.

        Returns:
            Scalar loss (reduction "mean"/"sum") or ``(N,)`` tensor ("none").
        """
        num_classes = logits.size(1)

        if self.use_softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)

        if target.dim() == 1:
            target_oh = F.one_hot(target, num_classes=num_classes).float()
        else:
            target_oh = target.float()

        if self.to_onehot_y and target.dim() == 1:
            target_oh = F.one_hot(target, num_classes=num_classes).float()

        pt = (probs * target_oh).sum(dim=1)
        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        loss = -focal_weight * torch.log(pt.clamp(min=1e-8))

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
