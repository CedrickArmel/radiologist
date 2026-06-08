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


def test_focal_loss_mean_reduction_returns_finite_scalar(batch):
    logits, targets = batch
    loss_fn = FocalLoss(gamma=2.0, to_onehot_y=True, reduction="mean")
    result = loss_fn(logits, targets)
    assert result.shape == torch.Size([])
    assert result.numel() == 1
    assert torch.isfinite(result)


def test_focal_loss_none_reduction_returns_per_sample_tensor(batch):
    logits, targets = batch
    loss_fn = FocalLoss(gamma=2.0, to_onehot_y=True, reduction="none")
    result = loss_fn(logits, targets)
    assert result.shape == (8,)


def test_focal_loss_sum_reduction_returns_scalar(batch):
    logits, targets = batch
    loss_fn = FocalLoss(gamma=2.0, to_onehot_y=True, reduction="sum")
    result = loss_fn(logits, targets)
    assert result.shape == torch.Size([])


def test_focal_loss_raises_value_error_for_invalid_reduction():
    with pytest.raises(ValueError, match="reduction"):
        FocalLoss(gamma=2.0, reduction="invalid")


def test_focal_loss_with_integer_targets_requires_to_onehot_y_true(batch):
    logits, targets = batch
    loss_fn = FocalLoss(gamma=2.0, to_onehot_y=True, reduction="mean")
    result = loss_fn(logits, targets)
    assert torch.isfinite(result)


def test_focal_loss_to_onehot_y_false_does_not_one_hot_integer_targets():
    """to_onehot_y=False must NOT one-hot-encode integer targets.

    With a single sample where logits strongly prefer class 0 and the target
    is class index 1 (the wrong class), the loss must be high. If one-hot
    conversion were applied incorrectly, the loss would be computed against
    the class-0 probability instead of the class-1 raw probability, yielding
    a different (lower) value. We verify the raw-probability path is taken by
    checking the loss differs from the to_onehot_y=True path.
    """
    torch.manual_seed(42)
    logits = torch.tensor([[5.0, -5.0]])  # strongly prefers class 0
    target_int = torch.tensor([1])  # class index 1

    loss_false = FocalLoss(gamma=2.0, to_onehot_y=False, reduction="mean")
    loss_true = FocalLoss(gamma=2.0, to_onehot_y=True, reduction="mean")

    result_int_false = loss_false(logits, target_int)
    result_onehot_true = loss_true(logits, target_int)

    # AC1: to_onehot_y=False with int targets must NOT equal to_onehot_y=True
    # (they consume different tensors: raw prob vs one-hot-weighted prob)
    assert not torch.isclose(result_int_false, result_onehot_true), (
        "to_onehot_y=False should produce a different loss than to_onehot_y=True "
        "when given integer targets, but they were equal — one-hot gate is broken"
    )


def test_focal_loss_to_onehot_y_true_matches_pre_built_one_hot():
    """to_onehot_y=True with integer targets must equal to_onehot_y=False with pre-built one-hot."""
    torch.manual_seed(42)
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    target_int = torch.tensor([0])
    target_onehot = torch.tensor([[1.0, 0.0, 0.0]])

    loss_true = FocalLoss(gamma=2.0, to_onehot_y=True, reduction="mean")
    loss_false = FocalLoss(gamma=2.0, to_onehot_y=False, reduction="mean")

    result_via_int = loss_true(logits, target_int)
    result_via_onehot = loss_false(logits, target_onehot)

    assert torch.isclose(result_via_int, result_via_onehot), (
        f"to_onehot_y=True(int) = {result_via_int.item():.6f} but "
        f"to_onehot_y=False(onehot) = {result_via_onehot.item():.6f}"
    )


def test_focal_loss_to_onehot_y_false_passes_through_one_hot_targets():
    """to_onehot_y=False with pre-built one-hot targets must not re-encode them."""
    torch.manual_seed(42)
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    target_onehot_class0 = torch.tensor([[1.0, 0.0, 0.0]])
    target_onehot_class1 = torch.tensor([[0.0, 1.0, 0.0]])

    loss_fn = FocalLoss(gamma=2.0, to_onehot_y=False, reduction="mean")

    result_class0 = loss_fn(logits, target_onehot_class0)
    result_class1 = loss_fn(logits, target_onehot_class1)

    # Results must differ because the one-hot targets are different
    assert not torch.isclose(result_class0, result_class1), (
        "to_onehot_y=False must use one-hot targets as-is; "
        "different one-hot targets must yield different losses"
    )
