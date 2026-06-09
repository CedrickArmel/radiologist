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

from unittest.mock import MagicMock, patch

import pytest
import torch

from radiologist.core import BestMetricCallback, WandbDefineSummaryCallback

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(metrics=None, loggers=None, is_global_zero=True):
    """Return a minimal trainer stub with callback_metrics and loggers."""
    trainer = MagicMock()
    trainer.callback_metrics = metrics if metrics is not None else {}
    trainer.loggers = loggers if loggers is not None else []
    trainer.is_global_zero = is_global_zero
    return trainer


# ---------------------------------------------------------------------------
# BestMetricCallback
# ---------------------------------------------------------------------------


def test_best_metric_callback_raises_for_invalid_mode():
    with pytest.raises(ValueError):
        BestMetricCallback(monitor="val_score", mode="bad")


def test_best_metric_callback_tracks_best_value_across_improving_epochs():
    cb = BestMetricCallback(monitor="val_score", mode="max")
    pl_module = MagicMock()

    trainer = _make_trainer(metrics={"val_score": torch.tensor(0.6)})
    cb.on_validation_epoch_end(trainer, pl_module)

    trainer2 = _make_trainer(metrics={"val_score": torch.tensor(0.8)})
    cb.on_validation_epoch_end(trainer2, pl_module)

    assert trainer2.callback_metrics["best_val_score"] == pytest.approx(0.8)


def test_best_metric_callback_does_not_update_on_non_improving_epoch():
    cb = BestMetricCallback(monitor="val_score", mode="max")
    pl_module = MagicMock()

    trainer1 = _make_trainer(metrics={"val_score": torch.tensor(0.8)})
    cb.on_validation_epoch_end(trainer1, pl_module)

    trainer2 = _make_trainer(metrics={"val_score": torch.tensor(0.5)})
    cb.on_validation_epoch_end(trainer2, pl_module)

    assert trainer2.callback_metrics["best_val_score"] == pytest.approx(0.8)


def test_best_metric_callback_noop_when_monitor_absent():
    cb = BestMetricCallback(monitor="val_score", mode="max")
    pl_module = MagicMock()

    trainer = _make_trainer(metrics={})
    cb.on_validation_epoch_end(trainer, pl_module)  # must not raise


def test_best_metric_callback_logs_to_every_logger():
    logger_a = MagicMock()
    logger_b = MagicMock()
    cb = BestMetricCallback(monitor="val_score", mode="max")
    pl_module = MagicMock()

    trainer = _make_trainer(
        metrics={"val_score": torch.tensor(0.7)},
        loggers=[logger_a, logger_b],
    )
    cb.on_validation_epoch_end(trainer, pl_module)

    for logger in (logger_a, logger_b):
        logger.log_metrics.assert_called_once()
        logged = logger.log_metrics.call_args[0][0]
        assert "best_val_score" in logged
        assert logged["best_val_score"] == pytest.approx(0.7)


def test_best_metric_callback_skips_logger_write_on_non_global_zero_rank():
    logger = MagicMock()
    cb = BestMetricCallback(monitor="val_score", mode="max")
    pl_module = MagicMock()

    trainer = _make_trainer(
        metrics={"val_score": torch.tensor(0.9)},
        loggers=[logger],
        is_global_zero=False,
    )
    cb.on_validation_epoch_end(trainer, pl_module)

    logger.log_metrics.assert_not_called()
    assert trainer.callback_metrics["best_val_score"] == pytest.approx(0.9)


def test_best_metric_callback_writes_logger_exactly_once_per_logger_on_global_zero():
    logger_a = MagicMock()
    logger_b = MagicMock()
    cb = BestMetricCallback(monitor="val_score", mode="max")
    pl_module = MagicMock()

    trainer = _make_trainer(
        metrics={"val_score": torch.tensor(0.7)},
        loggers=[logger_a, logger_b],
        is_global_zero=True,
    )
    cb.on_validation_epoch_end(trainer, pl_module)

    for logger in (logger_a, logger_b):
        logger.log_metrics.assert_called_once()
        logged = logger.log_metrics.call_args[0][0]
        assert logged["best_val_score"] == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# WandbDefineSummaryCallback
# ---------------------------------------------------------------------------


def test_wandb_define_summary_noop_when_wandb_absent():
    cb = WandbDefineSummaryCallback(monitor="val_score", mode="max")
    trainer = _make_trainer()
    pl_module = MagicMock()
    cb.on_fit_start(
        trainer, pl_module
    )  # wandb.run is None in test env — must not raise


def test_wandb_define_summary_calls_define_metric_when_run_active():
    import wandb

    cb = WandbDefineSummaryCallback(monitor="val_score", mode="max")
    trainer = _make_trainer()
    pl_module = MagicMock()

    fake_run = MagicMock()
    with patch.object(wandb, "run", fake_run):
        cb.on_fit_start(trainer, pl_module)

    fake_run.define_metric.assert_any_call("val_score", summary="max")
    fake_run.define_metric.assert_any_call("best_val_score", summary="max")


def test_wandb_define_summary_noop_when_run_is_none():
    cb = WandbDefineSummaryCallback(monitor="val_score", mode="max")
    trainer = _make_trainer()
    pl_module = MagicMock()
    cb.on_fit_start(
        trainer, pl_module
    )  # wandb.run is None in test env — must not raise
