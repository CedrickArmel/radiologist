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

import sys
import types
from functools import partial

import pytest
import torch
from omegaconf import OmegaConf
from torch.optim import SGD
from torch.optim.lr_scheduler import ConstantLR, SequentialLR

from radiologist.utils.ml import (
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    sequential_scheduler,
    task_wrapper,
)

# ---------------------------------------------------------------------------
# instantiate_callbacks
# ---------------------------------------------------------------------------


def test_instantiate_callbacks_returns_none_for_none_cfg() -> None:
    assert instantiate_callbacks(None) is None


def test_instantiate_callbacks_returns_none_for_empty_dictconfig() -> None:
    cfg = OmegaConf.create({})
    assert instantiate_callbacks(cfg) is None


def test_instantiate_callbacks_returns_list_for_valid_cfg() -> None:
    from lightning.pytorch.callbacks import ModelCheckpoint

    cfg = OmegaConf.create(
        {"ckpt": {"_target_": "lightning.pytorch.callbacks.ModelCheckpoint"}}
    )
    result = instantiate_callbacks(cfg)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], ModelCheckpoint)


def test_instantiate_callbacks_raises_type_error_for_non_dictconfig() -> None:
    with pytest.raises(TypeError):
        instantiate_callbacks("not a DictConfig")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# instantiate_loggers
# ---------------------------------------------------------------------------


def test_instantiate_loggers_returns_none_for_none_cfg() -> None:
    assert instantiate_loggers(None) is None


def test_instantiate_loggers_returns_none_for_empty_dictconfig() -> None:
    cfg = OmegaConf.create({})
    assert instantiate_loggers(cfg) is None


def test_instantiate_loggers_returns_list_for_valid_cfg() -> None:
    from lightning.pytorch.loggers import CSVLogger

    cfg = OmegaConf.create(
        {
            "csv": {
                "_target_": "lightning.pytorch.loggers.CSVLogger",
                "save_dir": "/tmp",
            }
        }
    )
    result = instantiate_loggers(cfg)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], CSVLogger)


# ---------------------------------------------------------------------------
# sequential_scheduler
# ---------------------------------------------------------------------------


def test_sequential_scheduler_returns_sequential_lr() -> None:
    model = torch.nn.Linear(2, 2)
    opt = SGD(model.parameters(), lr=0.01)
    sched_a = partial(ConstantLR, factor=1.0, total_iters=500)
    sched_b = partial(ConstantLR, factor=0.1, total_iters=500)
    result = sequential_scheduler(opt, [sched_a, sched_b], milestones=[500])
    assert isinstance(result, SequentialLR)


# ---------------------------------------------------------------------------
# get_metric_value
# ---------------------------------------------------------------------------


def test_get_metric_value_returns_zero_for_falsy_name() -> None:
    assert get_metric_value({}, None) == 0
    assert get_metric_value({"val_loss": torch.tensor(0.5)}, "") == 0


def test_get_metric_value_returns_float_for_existing_key() -> None:
    metrics = {"val_loss": torch.tensor(0.5)}
    result = get_metric_value(metrics, "val_loss")
    assert result == pytest.approx(0.5)


def test_get_metric_value_raises_for_missing_key() -> None:
    with pytest.raises(Exception):
        get_metric_value({}, "val_loss")


# ---------------------------------------------------------------------------
# task_wrapper
# ---------------------------------------------------------------------------


def test_task_wrapper_reraises_exception_and_finalizes_wandb() -> None:
    stub_wandb = types.ModuleType("wandb")
    finish_called = []

    def fake_finish() -> None:
        finish_called.append(True)

    stub_wandb.finish = fake_finish  # type: ignore[attr-defined]

    original = sys.modules.get("wandb")
    sys.modules["wandb"] = stub_wandb
    try:

        @task_wrapper
        def failing_task(cfg: object) -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            cfg = OmegaConf.create({"paths": {"output_dir": "logging/path"}})
            failing_task(cfg)

        assert finish_called, "wandb.finish was not called"
    finally:
        if original is None:
            del sys.modules["wandb"]
        else:
            sys.modules["wandb"] = original


# ---------------------------------------------------------------------------
# public API re-export
# ---------------------------------------------------------------------------


def test_public_api_imports_resolve() -> None:
    from radiologist.utils.ml import (  # noqa: F401
        extras,
        get_metric_value,
        instantiate_callbacks,
        instantiate_loggers,
        sequential_scheduler,
        task_wrapper,
    )


# ---------------------------------------------------------------------------
# print_config_tree / enforce_tags re-export
# ---------------------------------------------------------------------------


def test_print_config_tree_and_enforce_tags_importable_from_ml() -> None:
    from radiologist.utils.ml import enforce_tags, print_config_tree  # noqa: F401

    assert callable(print_config_tree)
    assert callable(enforce_tags)


# ---------------------------------------------------------------------------
# log_hyperparameters — "module" key
# ---------------------------------------------------------------------------


class _FakeLogger:
    def __init__(self) -> None:
        self.logged: dict = {}

    def log_hyperparams(self, params: dict) -> None:
        self.logged.update(params)


class _FakeTrainer:
    def __init__(self) -> None:
        self.logger = _FakeLogger()
        self.loggers = [_FakeLogger(), _FakeLogger()]


class _FakeModule:
    pass


def test_log_hyperparameters_logs_module_name_from_module_key() -> None:
    from radiologist.utils.ml import log_hyperparameters

    trainer = _FakeTrainer()
    module = _FakeModule()
    log_hyperparameters(
        {"cfg": OmegaConf.create({}), "module": module, "trainer": trainer}
    )
    for lgr in trainer.loggers:
        assert lgr.logged.get("module") == "_FakeModule"


def test_log_hyperparameters_does_not_raise_when_module_key_absent() -> None:
    from radiologist.utils.ml import log_hyperparameters

    trainer = _FakeTrainer()
    log_hyperparameters({"cfg": OmegaConf.create({}), "trainer": trainer})
