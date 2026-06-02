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

import pytest
import torch

from radiologist.core import FocalLoss


@pytest.fixture
def batch():
    torch.manual_seed(0)
    logits = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    return logits, targets


def test_focal_loss_mean_reduction_returns_scalar(batch):
    logits, targets = batch
    loss_fn = FocalLoss(gamma=2.0, reduction="mean")
    result = loss_fn(logits, targets)
    assert result.shape == torch.Size([])
    assert result.numel() == 1


def test_focal_loss_mean_reduction_is_finite(batch):
    logits, targets = batch
    loss_fn = FocalLoss(gamma=2.0, reduction="mean")
    result = loss_fn(logits, targets)
    assert torch.isfinite(result)


def test_focal_loss_none_reduction_returns_per_sample_tensor(batch):
    logits, targets = batch
    loss_fn = FocalLoss(gamma=2.0, reduction="none")
    result = loss_fn(logits, targets)
    assert result.shape == (8,)


def test_focal_loss_sum_reduction_returns_scalar(batch):
    logits, targets = batch
    loss_fn = FocalLoss(gamma=2.0, reduction="sum")
    result = loss_fn(logits, targets)
    assert result.shape == torch.Size([])


def test_focal_loss_raises_value_error_for_invalid_reduction():
    with pytest.raises(ValueError, match="reduction"):
        FocalLoss(gamma=2.0, reduction="invalid")


def test_focal_loss_with_integer_targets_class_index(batch):
    logits, targets = batch
    loss_fn = FocalLoss(gamma=2.0, to_onehot_y=False, reduction="mean")
    result = loss_fn(logits, targets)
    assert torch.isfinite(result)
