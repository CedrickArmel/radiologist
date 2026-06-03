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

from functools import partial

import pytest
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassFBetaScore

from radiologist.core import FocalLoss, LModule

NUM_CLASSES = 2
IN_FEATURES = 4
BATCH_SIZE = 4


@pytest.fixture
def net():
    return nn.Linear(IN_FEATURES, NUM_CLASSES)


@pytest.fixture
def loss_fn():
    return FocalLoss(gamma=2.0, reduction="mean", to_onehot_y=True)


@pytest.fixture
def metric_partial():
    return partial(MulticlassFBetaScore, num_classes=NUM_CLASSES, beta=1.0)


@pytest.fixture
def optimizer_partial():
    return partial(torch.optim.AdamW, lr=1e-3)


@pytest.fixture
def module(net, loss_fn, metric_partial, optimizer_partial):  # noqa: F811
    return LModule(
        net=net,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
    )


@pytest.fixture
def fake_batch():
    torch.manual_seed(42)
    return {
        "input": torch.randn(BATCH_SIZE, IN_FEATURES),
        "target": torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)),
        "key": ["a", "b", "c", "d"],
    }


def test_lmodule_instantiates_without_error(module):
    assert module is not None


def test_lmodule_forward_returns_correct_shape(module, fake_batch):
    x = fake_batch["input"]
    output = module(x)
    assert output.shape == (BATCH_SIZE, NUM_CLASSES)


def test_training_step_runs_without_error(module, fake_batch):
    loss = module.training_step(fake_batch, batch_idx=0)
    assert loss is not None
    assert torch.isfinite(loss)


def test_training_step_loss_is_scalar(module, fake_batch):
    loss = module.training_step(fake_batch, batch_idx=0)
    assert loss.shape == torch.Size([])


def test_validation_step_returns_logits(module, fake_batch):
    logits = module.validation_step(fake_batch, batch_idx=0)
    assert logits is not None
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_test_step_returns_logits(module, fake_batch):
    logits = module.test_step(fake_batch, batch_idx=0)
    assert logits is not None
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_test_step_appends_preds_to_output(module, fake_batch):
    module.test_step(fake_batch, batch_idx=0)
    assert hasattr(module, "output")
    assert len(module.output) == 1


def test_configure_optimizers_returns_dict_with_optimizer(module):
    result = module.configure_optimizers()
    assert "optimizer" in result


def test_configure_optimizers_no_scheduler_returns_optimizer_only(module):
    result = module.configure_optimizers()
    assert "lr_scheduler" not in result


def test_configure_optimizers_with_scheduler_returns_scheduler_dict(
    net, loss_fn, metric_partial
):
    optimizer_partial = partial(torch.optim.AdamW, lr=1e-3)
    scheduler_partial = partial(torch.optim.lr_scheduler.StepLR, step_size=1, gamma=0.9)
    mod = LModule(
        net=net,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
        scheduler=scheduler_partial,
    )
    result = mod.configure_optimizers()
    assert "optimizer" in result
    assert "lr_scheduler" in result
    assert result["lr_scheduler"]["interval"] == "step"


def test_save_hyperparameters_excludes_net_and_loss(module):
    assert len(module.hparams) > 0
    assert "net" not in module.hparams
    assert "loss" not in module.hparams


def test_on_validation_epoch_end_tracks_best_val_score(module, fake_batch):
    module.validation_step(fake_batch, batch_idx=0)
    module.on_validation_epoch_end()
    assert hasattr(module, "val_score_best")
