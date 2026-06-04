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
from unittest.mock import MagicMock

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


# ---------------------------------------------------------------------------
# AC: val_score_best removed — BestMetricCallback is the single source of truth
# ---------------------------------------------------------------------------


def test_lmodule_does_not_expose_val_score_best(module):
    assert not hasattr(module, "val_score_best")


def test_lmodule_does_not_define_on_validation_epoch_end(module):
    assert "on_validation_epoch_end" not in type(module).__dict__


def test_best_metric_callback_produces_best_val_score_in_callback_metrics(
    module, fake_batch
):
    from unittest.mock import MagicMock

    from radiologist.core import BestMetricCallback

    cb = BestMetricCallback(monitor="val_score", mode="max")
    trainer = MagicMock()
    trainer.callback_metrics = {}
    trainer.loggers = []

    module.validation_step(fake_batch, batch_idx=0)
    val_score_value = module.val_score.compute()
    trainer.callback_metrics["val_score"] = val_score_value

    cb.on_validation_epoch_end(trainer, module)

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
        nn.Linear(8, NUM_CLASSES),  # index 2  ← last Linear
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


def test_training_step_returns_finite_scalar_loss(module, fake_batch):
    loss = module.training_step(fake_batch, batch_idx=0)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# AC: shared step — validation_step and test_step return correct logit shape
# ---------------------------------------------------------------------------


def test_validation_step_returns_logits_correct_shape(module, fake_batch):
    logits = module.validation_step(fake_batch, batch_idx=0)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


def test_test_step_returns_logits_correct_shape(module, fake_batch):
    logits = module.test_step(fake_batch, batch_idx=0)
    assert logits.shape == (BATCH_SIZE, NUM_CLASSES)


# ---------------------------------------------------------------------------
# AC: on_test_epoch_end saves concatenated preds to log_dir
# ---------------------------------------------------------------------------


def test_on_test_epoch_end_saves_preds_file(module, fake_batch, tmp_path):
    module.test_step(fake_batch, batch_idx=0)
    module.test_step(fake_batch, batch_idx=1)

    mock_trainer = MagicMock()
    mock_trainer.log_dir = str(tmp_path)
    mock_trainer.global_rank = 0
    module._trainer = mock_trainer

    module.on_test_epoch_end()

    expected_path = tmp_path / "preds-rank0.pt"
    assert expected_path.exists()
    saved = torch.load(str(expected_path), weights_only=True)
    assert saved.shape == (BATCH_SIZE * 2,)


def test_on_test_epoch_end_is_noop_when_log_dir_is_none(module, fake_batch):
    module.test_step(fake_batch, batch_idx=0)

    mock_trainer = MagicMock()
    mock_trainer.log_dir = None
    mock_trainer.global_rank = 0
    module._trainer = mock_trainer

    module.on_test_epoch_end()  # must not raise


# ---------------------------------------------------------------------------
# AC: grad-norm hook — on_before_optimizer_step always fires
# ---------------------------------------------------------------------------


def test_on_before_optimizer_step_logs_grad_norm_without_clipping(module, fake_batch):
    """on_before_optimizer_step must log grad_norm on every optimizer step."""
    logged: dict = {}
    module.log = lambda name, value, **kw: logged.update({name: value})

    module.training_step(fake_batch, batch_idx=0)
    optimizer = module.configure_optimizers()["optimizer"]
    module.on_before_optimizer_step(optimizer)

    assert "grad_norm" in logged


def test_grad_norm_value_is_non_negative(module, fake_batch):
    """grad_norm logged by on_before_optimizer_step must be >= 0."""
    logged: dict = {}
    module.log = lambda name, value, **kw: logged.update({name: value})

    module.training_step(fake_batch, batch_idx=0)
    optimizer = module.configure_optimizers()["optimizer"]
    module.on_before_optimizer_step(optimizer)

    assert logged["grad_norm"] >= 0


# ---------------------------------------------------------------------------
# AC: configure_gradient_clipping logs grad_norm_post_clip, not weight_norm_post_clip
# ---------------------------------------------------------------------------


def _make_mock_trainer(clip_val: float = 1.0) -> MagicMock:
    """Return a minimal Trainer mock that satisfies clip_gradients internals."""
    mock_trainer = MagicMock()
    mock_trainer.gradient_clip_val = clip_val
    mock_trainer.gradient_clip_algorithm = "norm"
    return mock_trainer


def test_configure_gradient_clipping_logs_grad_norm_post_clip(module, fake_batch):
    """configure_gradient_clipping must log grad_norm_post_clip after clipping."""
    logged: dict = {}
    module.log = lambda name, value, **kw: logged.update({name: value})
    module._trainer = _make_mock_trainer(clip_val=1.0)

    module.training_step(fake_batch, batch_idx=0)
    optimizer = module.configure_optimizers()["optimizer"]
    module.configure_gradient_clipping(
        optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
    )

    assert "grad_norm_post_clip" in logged


def test_grad_norm_post_clip_is_at_most_clip_threshold(module, fake_batch):
    """Post-clip grad norm must be <= gradient_clip_val for norm-based clipping."""
    logged: dict = {}
    module.log = lambda name, value, **kw: logged.update({name: value})
    clip_val = 0.1
    module._trainer = _make_mock_trainer(clip_val=clip_val)

    module.training_step(fake_batch, batch_idx=0)
    optimizer = module.configure_optimizers()["optimizer"]
    module.configure_gradient_clipping(
        optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm="norm"
    )

    assert float(logged["grad_norm_post_clip"]) <= clip_val + 1e-6


def test_configure_gradient_clipping_does_not_log_weight_norm_post_clip(
    module, fake_batch
):
    """configure_gradient_clipping must NOT log weight_norm_post_clip."""
    logged: dict = {}
    module.log = lambda name, value, **kw: logged.update({name: value})
    module._trainer = _make_mock_trainer(clip_val=1.0)

    module.training_step(fake_batch, batch_idx=0)
    optimizer = module.configure_optimizers()["optimizer"]
    module.configure_gradient_clipping(
        optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
    )

    assert "weight_norm_post_clip" not in logged


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
# AC: Bug A — grad/weight norm tensors must use self.device, not CPU
# ---------------------------------------------------------------------------


def test_grad_norm_tensor_lives_on_model_device(module, fake_batch):
    """_get_grad_norm must return a tensor on the same device as model params."""
    module.training_step(fake_batch, batch_idx=0)
    norm = module._get_grad_norm()
    param_device = next(module.parameters()).device
    assert norm.device == param_device


def test_weight_norm_tensor_lives_on_model_device(module, fake_batch):
    """_get_weight_norm must return a tensor on the same device as model params."""
    norm = module._get_weight_norm()
    param_device = next(module.parameters()).device
    assert norm.device == param_device


# ---------------------------------------------------------------------------
# AC: Bug B — on_test_epoch_end clears output buffer between runs
# ---------------------------------------------------------------------------


def test_second_test_run_yields_single_run_length(module, fake_batch, tmp_path):
    """Calling on_test_epoch_end twice must not accumulate preds across runs."""
    mock_trainer = MagicMock()
    mock_trainer.log_dir = str(tmp_path)
    mock_trainer.global_rank = 0
    module._trainer = mock_trainer

    # First run
    module.test_step(fake_batch, batch_idx=0)
    module.on_test_epoch_end()

    # Second run — buffer should have been cleared after first run
    module.test_step(fake_batch, batch_idx=0)
    module.on_test_epoch_end()

    saved = torch.load(str(tmp_path / "preds-rank0.pt"), weights_only=True)
    assert saved.shape == (BATCH_SIZE,)


def test_on_test_epoch_end_with_empty_output_does_not_raise(module, tmp_path):
    """on_test_epoch_end with no test_step calls must complete without error."""
    mock_trainer = MagicMock()
    mock_trainer.log_dir = str(tmp_path)
    mock_trainer.global_rank = 0
    module._trainer = mock_trainer

    module.on_test_epoch_end()  # no test_step was called — output is empty
