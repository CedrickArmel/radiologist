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

import builtins
import importlib
from unittest.mock import patch

import lightning as L
import pytest
import torch

from radiologist.core import OnnxExportCallback, WandbDefineSummaryCallback
from radiologist.core.callbacks import attribution as attribution_mod
from radiologist.core.callbacks import onnx_export as onnx_export_mod
from radiologist.core.callbacks import wandb_summary as wandb_summary_mod

_GUARDED_MODULES = [attribution_mod, onnx_export_mod, wandb_summary_mod]


@pytest.fixture
def wandb_import_blocked():
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("simulated missing wandb")
        return real_import(name, *args, **kwargs)

    try:
        with patch("builtins.__import__", side_effect=fake_import):
            for mod in _GUARDED_MODULES:
                importlib.reload(mod)
        yield
    finally:
        for mod in _GUARDED_MODULES:
            importlib.reload(mod)


def test_callback_modules_reload_succeeds_when_wandb_unavailable(
    wandb_import_blocked,
):
    for mod in _GUARDED_MODULES:
        assert mod.wandb is None


def test_wandb_summary_callback_noop_when_wandb_sentinel_none(lmodule, dm):
    with patch.object(wandb_summary_mod, "wandb", None, create=True):
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


def test_onnx_export_callback_noop_when_wandb_sentinel_none(lmodule, dm):
    with patch.object(onnx_export_mod, "wandb", None, create=True):
        cb = OnnxExportCallback(
            input_shape=(1, 3, 8, 8),
            classes=["healthy", "sick"],
            cam_target_layer="2",
        )
        trainer = L.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[cb],
            logger=False,
        )
        trainer.fit(lmodule, datamodule=dm)


def test_attribution_log_gallery_noop_when_wandb_sentinel_none():
    from radiologist.core.callbacks.attribution import AttributionCallback, _Panel

    cb = AttributionCallback(
        target_layer="0", every_n_val_epochs=1, output_subdir="attributions"
    )
    panel = _Panel(
        key="sample_0",
        image=torch.zeros(3, 4, 4),
        caption="caption",
        filename="sample_0.png",
    )

    with patch.object(attribution_mod, "wandb", None, create=True):
        cb._log_gallery([panel], stage="val", step=0)
