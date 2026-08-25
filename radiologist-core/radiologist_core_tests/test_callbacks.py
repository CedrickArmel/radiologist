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

import lightning as L
import pytest

from radiologist.core import BestMetricCallback, WandbDefineSummaryCallback

# ---------------------------------------------------------------------------
# BestMetricCallback — constructor validation (no trainer needed)
# ---------------------------------------------------------------------------


def test_best_metric_callback_raises_for_invalid_mode():
    with pytest.raises(ValueError):
        BestMetricCallback(monitor="val_score", mode="bad")


# ---------------------------------------------------------------------------
# BestMetricCallback — real trainer fit integration
# ---------------------------------------------------------------------------


def test_best_metric_written_to_callback_metrics(lmodule, dm):
    callback = BestMetricCallback(monitor="val_score", mode="max")
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
        logger=False,
    )
    trainer.fit(lmodule, datamodule=dm)
    assert "best_val_score" in trainer.callback_metrics


def test_best_metric_callback_best_val_score_is_non_negative(lmodule, dm):
    callback = BestMetricCallback(monitor="val_score", mode="max")
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
        logger=False,
    )
    trainer.fit(lmodule, datamodule=dm)
    assert float(trainer.callback_metrics["best_val_score"]) >= 0.0


def test_best_metric_callback_noop_when_monitor_absent(lmodule, dm):
    callback = BestMetricCallback(monitor="nonexistent_metric", mode="max")
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
        logger=False,
    )
    trainer.fit(lmodule, datamodule=dm)
    assert "best_nonexistent_metric" not in trainer.callback_metrics


def test_best_metric_callback_logs_to_wandb_when_run_active(lmodule, dm):
    import wandb

    from radiologist.core import WandbDefineSummaryCallback

    fake_run = MagicMock()
    best_cb = BestMetricCallback(monitor="val_score", mode="max")
    summary_cb = WandbDefineSummaryCallback(monitor="val_score", mode="max")
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[best_cb, summary_cb],
        logger=False,
    )
    with patch.object(wandb, "run", fake_run):
        trainer.fit(lmodule, datamodule=dm)

    assert "best_val_score" in trainer.callback_metrics


# ---------------------------------------------------------------------------
# WandbDefineSummaryCallback — real trainer fit integration
# ---------------------------------------------------------------------------


def test_wandb_define_summary_noop_when_wandb_absent(lmodule, dm):
    cb = WandbDefineSummaryCallback(monitor="val_score", mode="max")
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[cb],
        logger=False,
    )
    trainer.fit(lmodule, datamodule=dm)


def test_wandb_summary_defines_metrics_when_run_active(lmodule, dm):
    import wandb

    fake_run = MagicMock()
    cb = WandbDefineSummaryCallback(monitor="val_score", mode="max")
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[cb],
        logger=False,
    )
    with patch.object(wandb, "run", fake_run):
        trainer.fit(lmodule, datamodule=dm)
    fake_run.define_metric.assert_called()
