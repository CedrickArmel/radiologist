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


# ---------------------------------------------------------------------------
# AC: training_step returns a finite scalar loss
# ---------------------------------------------------------------------------


def test_training_step_returns_finite_scalar_loss(module, fake_batch):
    loss = module.training_step(fake_batch, batch_idx=0)
    assert loss is not None
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# AC: validation_step returns logits of the correct shape
# ---------------------------------------------------------------------------


def test_validation_step_returns_logits_correct_shape(module, fake_batch):
    logits = module.validation_step(fake_batch, batch_idx=0)
    assert logits is not None
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


# ---------------------------------------------------------------------------
# AC: test_step returns logits of the correct shape and appends preds
# ---------------------------------------------------------------------------


def test_test_step_returns_correct_logits_and_appends_preds(module, fake_batch):
    logits = module.test_step(fake_batch, batch_idx=0)
    assert logits is not None
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)
    assert hasattr(module, "output")
    assert len(module.output) == 1


# ---------------------------------------------------------------------------
# AC: configure_optimizers — no-scheduler path returns optimizer only
# ---------------------------------------------------------------------------


def test_configure_optimizers_no_scheduler_returns_optimizer_only(module):
    result = module.configure_optimizers()
    assert "optimizer" in result
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


# ---------------------------------------------------------------------------
# AC: val_score_best removed — BestMetricCallback is the single source of truth
# ---------------------------------------------------------------------------


def test_best_metric_callback_produces_best_val_score_in_callback_metrics(lmodule, dm):
    import lightning as L

    from radiologist.core import BestMetricCallback

    cb = BestMetricCallback(monitor="val_score", mode="max")
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[cb],
        logger=False,
    )
    trainer.fit(lmodule, datamodule=dm)

    assert "best_val_score" in trainer.callback_metrics


def test_validation_step_still_logs_val_score(module, fake_batch):
    logged: dict = {}
    module.log = lambda name, value, **kw: logged.update({name: value})

    module.validation_step(fake_batch, batch_idx=0)

    assert "val_score" in logged


# ---------------------------------------------------------------------------
# Fixtures for setup / transfer-learning tests
# ---------------------------------------------------------------------------


@pytest.fixture
def small_net():
    """A tiny sequential net with a named submodule and a final Linear."""
    net = nn.Sequential(
        nn.Linear(IN_FEATURES, 8),  # index 0
        nn.ReLU(),  # index 1
        nn.Linear(8, NUM_CLASSES),  # index 2  <- last Linear
    )
    return net


@pytest.fixture
def module_scratch(small_net, loss_fn, metric_partial, optimizer_partial):
    """LModule with trainable_layers=None (from-scratch mode)."""
    return LModule(
        net=small_net,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
        trainable_layers=None,
    )


@pytest.fixture
def module_tl(small_net, loss_fn, metric_partial, optimizer_partial):
    """LModule with transfer-learning: only index 2 (last Linear) trainable."""
    return LModule(
        net=small_net,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
        trainable_layers={"2": None},
    )


@pytest.fixture
def module_tl_with_priors(small_net, loss_fn, metric_partial, optimizer_partial):
    """LModule with transfer-learning and class priors."""
    return LModule(
        net=small_net,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
        trainable_layers={"2": None},
        priors=[0.3, 0.7],
    )


# ---------------------------------------------------------------------------
# AC: setup(fit) from-scratch re-initialises all params and leaves them trainable
# ---------------------------------------------------------------------------


def test_setup_from_scratch_all_params_remain_trainable(module_scratch):
    module_scratch.setup("fit")
    all_trainable = all(p.requires_grad for p in module_scratch.net.parameters())
    assert all_trainable


def test_setup_from_scratch_reinitialises_weights(
    small_net, loss_fn, metric_partial, optimizer_partial
):
    torch.manual_seed(0)
    original_weight = small_net[0].weight.clone()

    torch.manual_seed(0)
    mod = LModule(
        net=small_net,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
        trainable_layers=None,
    )
    mod.setup("fit")
    assert not torch.equal(mod.net[0].weight, original_weight)


# ---------------------------------------------------------------------------
# AC: setup(fit) transfer-learning freezes all then selectively unfreezes
# ---------------------------------------------------------------------------


def test_setup_tl_only_named_layers_are_trainable(module_tl, small_net):
    module_tl.setup("fit")
    # layer 2 (last Linear) should be trainable
    for p in small_net[2].parameters():
        assert p.requires_grad
    # layers 0 and 1 should be frozen
    for p in small_net[0].parameters():
        assert not p.requires_grad


def test_setup_tl_none_value_unfreezes_whole_submodule(
    small_net, loss_fn, metric_partial, optimizer_partial
):
    """trainable_layers with None value unfreezes entire named submodule."""
    mod = LModule(
        net=small_net,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
        trainable_layers={"2": None},
    )
    mod.setup("fit")
    for p in small_net[2].parameters():
        assert p.requires_grad


def test_setup_tl_list_of_indices_unfreezes_only_those(
    loss_fn, metric_partial, optimizer_partial
):
    """trainable_layers with list of ints unfreezes only those submodule indices."""
    outer = nn.Sequential(
        nn.Sequential(nn.Linear(4, 4), nn.ReLU()),  # index 0
        nn.Sequential(nn.Linear(4, 2), nn.ReLU()),  # index 1
    )
    mod = LModule(
        net=outer,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
        trainable_layers={"": [1]},
    )
    mod.setup("fit")
    for p in outer[1].parameters():
        assert p.requires_grad
    for p in outer[0].parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# AC: priors bias initialisation
# ---------------------------------------------------------------------------


def test_setup_tl_with_priors_sets_last_linear_bias(module_tl_with_priors, small_net):
    module_tl_with_priors.setup("fit")
    expected = -torch.log(torch.tensor([0.3, 0.7], dtype=torch.float32))
    actual = small_net[2].bias.data
    assert torch.allclose(actual, expected, atol=1e-5)


def test_setup_tl_priors_length_mismatch_raises_value_error(
    small_net, loss_fn, metric_partial, optimizer_partial
):
    mod = LModule(
        net=small_net,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
        trainable_layers={"2": None},
        priors=[0.5, 0.3, 0.2],  # 3 priors vs 2 out_features
    )
    with pytest.raises(ValueError):
        mod.setup("fit")


def test_setup_tl_no_priors_leaves_last_linear_bias_unchanged(
    small_net, loss_fn, metric_partial, optimizer_partial
):
    original_bias = small_net[2].bias.data.clone()
    mod = LModule(
        net=small_net,
        loss=loss_fn,
        metric=metric_partial,
        optimizer=optimizer_partial,
        trainable_layers={"2": None},
        priors=None,
    )
    mod.setup("fit")
    assert torch.equal(small_net[2].bias.data, original_bias)


# ---------------------------------------------------------------------------
# AC: train_score removed
# ---------------------------------------------------------------------------


def test_training_step_does_not_log_train_score(module, fake_batch):
    assert not hasattr(module, "train_score")


# ---------------------------------------------------------------------------
# AC: on_test_epoch_end saves concatenated preds to log_dir
# ---------------------------------------------------------------------------


def test_on_test_epoch_end_saves_preds_file(lmodule, dm, tmp_path):
    import lightning as L

    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        logger=False,
    )
    trainer.test(lmodule, datamodule=dm)

    preds_files = list(tmp_path.rglob("preds-rank*.pt"))
    assert len(preds_files) == 1
    saved = torch.load(str(preds_files[0]), weights_only=True)
    assert saved.ndim == 1


# ---------------------------------------------------------------------------
# AC: grad-norm hook — on_before_optimizer_step always fires with a non-negative
#     norm tensor on the model device
# ---------------------------------------------------------------------------


def test_on_before_optimizer_step_logs_non_negative_grad_norm_on_model_device(
    module, fake_batch
):
    """on_before_optimizer_step must log a non-negative grad_norm on the model device."""
    logged: dict = {}
    module.log = lambda name, value, **kw: logged.update({name: value})

    module.training_step(fake_batch, batch_idx=0)
    optimizer = module.configure_optimizers()["optimizer"]
    module.on_before_optimizer_step(optimizer)

    assert "grad_norm" in logged
    assert logged["grad_norm"] >= 0
    param_device = next(module.parameters()).device
    norm = module._get_grad_norm()
    assert norm.device == param_device


# ---------------------------------------------------------------------------
# AC: configure_gradient_clipping logs grad_norm_post_clip, not weight_norm_post_clip
# ---------------------------------------------------------------------------


def test_configure_gradient_clipping_logs_post_clip_norm_in_callback_metrics(
    lmodule, dm
):
    """trainer.fit with gradient_clip_val must record grad_norm_post_clip
    in callback_metrics and must NOT record weight_norm_post_clip."""
    import lightning as L

    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        logger=False,
    )
    trainer.fit(lmodule, datamodule=dm)

    assert "grad_norm_post_clip" in trainer.callback_metrics
    assert "weight_norm_post_clip" not in trainer.callback_metrics


# ---------------------------------------------------------------------------
# AC: on_train_batch_end still logs weight_norm
# ---------------------------------------------------------------------------


def test_on_train_batch_end_logs_weight_norm(module, fake_batch):
    """on_train_batch_end must log weight_norm on every training batch."""
    logged: dict = {}
    module.log = lambda name, value, **kw: logged.update({name: value})

    module.on_train_batch_end(outputs=None, batch=fake_batch, batch_idx=0)

    assert "weight_norm" in logged


# ---------------------------------------------------------------------------
# AC: Bug A — weight norm tensor must use self.device, not CPU
# ---------------------------------------------------------------------------


def test_weight_norm_tensor_lives_on_model_device(module, fake_batch):
    """_get_weight_norm must return a tensor on the same device as model params."""
    norm = module._get_weight_norm()
    param_device = next(module.parameters()).device
    assert norm.device == param_device


# ---------------------------------------------------------------------------
# AC: Bug B — on_test_epoch_end clears output buffer between runs
# ---------------------------------------------------------------------------


def test_second_test_run_yields_single_run_length(lmodule, dm, tmp_path):
    """Calling trainer.test twice must not accumulate preds across runs."""
    import lightning as L

    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        logger=False,
    )
    trainer.test(lmodule, datamodule=dm)
    first_preds = list(tmp_path.rglob("preds-rank*.pt"))
    assert len(first_preds) == 1
    first_size = torch.load(str(first_preds[0]), weights_only=True).shape[0]

    trainer.test(lmodule, datamodule=dm)
    second_preds = list(tmp_path.rglob("preds-rank*.pt"))
    second_size = torch.load(str(second_preds[0]), weights_only=True).shape[0]

    assert second_size == first_size, "buffer must be cleared between test runs"


def test_on_test_epoch_end_with_empty_output_does_not_raise(lmodule, dm, tmp_path):
    """trainer.test with an empty output buffer must complete without error."""
    import lightning as L

    lmodule.output.clear()
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        logger=False,
    )
    trainer.test(lmodule, datamodule=dm)
