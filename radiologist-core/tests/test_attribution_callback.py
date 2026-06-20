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

from __future__ import annotations

import sys
import types
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock, patch

import lightning as L
import pytest
import torch
import torch.nn as nn
from torchmetrics.classification import (
    MulticlassFBetaScore,  # type: ignore[import-untyped]
)

from radiologist.core import FocalLoss, LModule

# ---------------------------------------------------------------------------
# Helpers — real LModule with a 1-channel net for attribution hooks
# ---------------------------------------------------------------------------


def _make_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 2),
    )


def _make_lmodule() -> LModule:
    net = _make_net()
    return LModule(
        net=net,
        loss=FocalLoss(),
        metric=partial(MulticlassFBetaScore, beta=1.0, num_classes=2),
        optimizer=partial(torch.optim.Adam, lr=1e-3),
    )


def _make_trainer(tmp_path: Path) -> L.Trainer:
    """Return a real trainer whose log_dir points to tmp_path."""
    return L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        logger=False,
    )


def _make_batch(n: int = 4) -> dict:
    """All samples share class 0 by default."""
    return {
        "input": torch.randn(n, 1, 8, 8),
        "target": torch.zeros(n, dtype=torch.long),
        "key": [f"sample_{i}" for i in range(n)],
    }


def _make_outputs(n: int = 4, n_classes: int = 2) -> torch.Tensor:
    return torch.randn(n, n_classes)


# ---------------------------------------------------------------------------
# AC: module imports cleanly even without captum
# ---------------------------------------------------------------------------


def test_attribution_callback_importable_without_captum():
    from radiologist.core import AttributionCallback  # noqa: F401

    assert AttributionCallback is not None


# ---------------------------------------------------------------------------
# AC: __init__ does not resolve target_layer (lazy)
# ---------------------------------------------------------------------------


def test_init_does_not_raise_for_invalid_target_layer():
    from radiologist.core import AttributionCallback

    cb = AttributionCallback(target_layer="nonexistent.layer.path")
    assert cb is not None


# ---------------------------------------------------------------------------
# AC: bad dot-path raises AttributeError at FIRST USE, not at init
# ---------------------------------------------------------------------------


def test_bad_target_layer_raises_attribute_error_at_first_use(tmp_path):
    fake_captum = types.ModuleType("captum")
    fake_attr = types.ModuleType("captum.attr")
    fake_captum.attr = fake_attr

    fake_attr.LayerGradCam = MagicMock()
    fake_attr.IntegratedGradients = MagicMock()

    with patch.dict(sys.modules, {"captum": fake_captum, "captum.attr": fake_attr}):
        import importlib

        import radiologist.core.callbacks.attribution as attr_mod

        importlib.reload(attr_mod)
        AttrCB = attr_mod.AttributionCallback

        cb = AttrCB(target_layer="bad.path.does.not.exist", every_n_val_epochs=1)
        trainer = _make_trainer(tmp_path)
        lm = _make_lmodule()
        batch = _make_batch()
        outputs = _make_outputs()

        with pytest.raises(AttributeError):
            cb.on_validation_batch_end(trainer, lm, outputs, batch, batch_idx=0)


# ---------------------------------------------------------------------------
# AC: batch_idx % every_n_batches != 0 → no-op (no files written)
# ---------------------------------------------------------------------------


def test_validation_wrong_batch_idx_writes_no_files(tmp_path):
    from radiologist.core import AttributionCallback

    cb = AttributionCallback(target_layer="0", every_n_val_epochs=1, every_n_batches=10)
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()
    batch = _make_batch()
    outputs = _make_outputs()

    cb.on_validation_batch_end(trainer, lm, outputs, batch, batch_idx=1)

    assert list(tmp_path.rglob("*.png")) == []


# ---------------------------------------------------------------------------
# AC: epoch % every_n_val_epochs != 0 → no-op even when step is right
# Epoch 0 matches every_n_val_epochs=1 (fires). Non-matching tested via batches.
# ---------------------------------------------------------------------------


def test_validation_fires_on_matching_epoch(tmp_path):
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", every_n_val_epochs=1, output_subdir="attributions")
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()
    batch = _make_batch()
    outputs = _make_outputs()

    with patch.dict(sys.modules, {"captum": fake_captum, "captum.attr": fake_attr}):
        cb.on_validation_batch_end(trainer, lm, outputs, batch, batch_idx=0)

    assert len(list((tmp_path / "attributions").glob("*.png"))) >= 1


# ---------------------------------------------------------------------------
# AC: captum absent → silent no-op, no files, no exception
# ---------------------------------------------------------------------------


def _reload_attribution_without_captum():
    import importlib

    import radiologist.core.callbacks.attribution as attr_mod

    with patch.dict(sys.modules, {"captum": None, "captum.attr": None}):
        importlib.reload(attr_mod)
        return attr_mod.AttributionCallback


def test_no_png_written_when_captum_absent(tmp_path):
    AttrCB = _reload_attribution_without_captum()

    cb = AttrCB(target_layer="0", every_n_val_epochs=1)
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()
    batch = _make_batch()
    outputs = _make_outputs()

    cb.on_validation_batch_end(trainer, lm, outputs, batch, batch_idx=0)

    assert list(tmp_path.rglob("*.png")) == []


def test_no_exception_when_captum_absent_test_batch(tmp_path):
    AttrCB = _reload_attribution_without_captum()

    cb = AttrCB(target_layer="0")
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()
    batch = _make_batch()
    outputs = _make_outputs()

    cb.on_test_batch_end(trainer, lm, outputs, batch, batch_idx=0)

    assert list(tmp_path.rglob("*.png")) == []


# ---------------------------------------------------------------------------
# Fake captum modules — batch-size-aware stubs
# ---------------------------------------------------------------------------


def _make_fake_captum_modules():
    fake_captum = types.ModuleType("captum")
    fake_attr = types.ModuleType("captum.attr")
    fake_captum.attr = fake_attr

    def _fake_gradcam_attr(inputs, target, **kwargs):
        return torch.ones(inputs.shape[0], 1, 8, 8)

    def _fake_ig_attr(inputs, target, **kwargs):
        return torch.ones(inputs.shape[0], inputs.shape[1], 8, 8)

    gradcam_instance = MagicMock()
    gradcam_instance.attribute = MagicMock(side_effect=_fake_gradcam_attr)
    ig_instance = MagicMock()
    ig_instance.attribute = MagicMock(side_effect=_fake_ig_attr)

    fake_attr.LayerGradCam = MagicMock(return_value=gradcam_instance)
    fake_attr.IntegratedGradients = MagicMock(return_value=ig_instance)
    return fake_captum, fake_attr


def _reload_attribution_with_captum(fake_captum, fake_attr):
    import importlib

    import radiologist.core.callbacks.attribution as attr_mod

    with patch.dict(sys.modules, {"captum": fake_captum, "captum.attr": fake_attr}):
        importlib.reload(attr_mod)
        return attr_mod.AttributionCallback


# ---------------------------------------------------------------------------
# AC: captum present → one PNG per class present in the batch
# ---------------------------------------------------------------------------


def test_png_files_written_with_captum_on_validation(tmp_path):
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", every_n_val_epochs=1, output_subdir="attributions")
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()
    batch = _make_batch(n=4)
    outputs = _make_outputs(n=4)

    with patch.dict(sys.modules, {"captum": fake_captum, "captum.attr": fake_attr}):
        cb.on_validation_batch_end(trainer, lm, outputs, batch, batch_idx=0)

    out_dir = tmp_path / "attributions"
    written = sorted(p.name for p in out_dir.glob("*.png"))

    assert len(written) == 1
    assert written[0].startswith("val-ep000-key-sample_")
    assert "true0" in written[0]


def test_png_filenames_are_deterministic_overwrite_on_rerun(tmp_path):
    """Second call at same step must produce the same filenames."""
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", every_n_val_epochs=1, output_subdir="attributions")
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()
    batch = _make_batch(n=2)
    outputs = _make_outputs(n=2)

    with patch.dict(sys.modules, {"captum": fake_captum, "captum.attr": fake_attr}):
        cb.on_validation_batch_end(trainer, lm, outputs, batch, batch_idx=0)
        first_files = set(p.name for p in (tmp_path / "attributions").glob("*.png"))
        cb.on_validation_batch_end(trainer, lm, outputs, batch, batch_idx=0)
        second_files = set(p.name for p in (tmp_path / "attributions").glob("*.png"))

    assert first_files == second_files, "second run produced different filenames"


def test_test_batch_end_writes_png_when_step_matches(tmp_path):
    """on_test_batch_end runs when global_step % every_n_batches == 0."""
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", output_subdir="attributions")
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()
    batch = _make_batch(n=2)
    outputs = _make_outputs(n=2)

    with patch.dict(sys.modules, {"captum": fake_captum, "captum.attr": fake_attr}):
        cb.on_test_batch_end(trainer, lm, outputs, batch, batch_idx=0)

    out_dir = tmp_path / "attributions"
    written = sorted(p.name for p in out_dir.glob("*.png"))

    assert len(written) >= 1
    assert any("test-ep000" in n for n in written)


def test_test_batch_end_skipped_when_batch_idx_does_not_match(tmp_path):
    """on_test_batch_end is a no-op when batch_idx % every_n_batches != 0."""
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", every_n_batches=10)
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()

    with patch.dict(sys.modules, {"captum": fake_captum, "captum.attr": fake_attr}):
        cb.on_test_batch_end(trainer, lm, _make_outputs(), _make_batch(), batch_idx=1)

    assert list(tmp_path.rglob("*.png")) == []


# ---------------------------------------------------------------------------
# AC: misclassified sample is preferred over correct sample for its class
# ---------------------------------------------------------------------------


def test_misclassified_sample_preferred_for_its_class(tmp_path):
    """For class 0, sample_1 (wrong pred) is selected over sample_0 (correct)."""
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", every_n_val_epochs=1, output_subdir="attributions")
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()

    batch = {
        "input": torch.randn(4, 1, 8, 8),
        "target": torch.zeros(4, dtype=torch.long),
        "key": ["sample_0", "sample_1", "sample_2", "sample_3"],
    }
    outputs = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0]])

    with patch.dict(sys.modules, {"captum": fake_captum, "captum.attr": fake_attr}):
        cb.on_validation_batch_end(trainer, lm, outputs, batch, batch_idx=0)

    written = [p.name for p in (tmp_path / "attributions").glob("*.png")]

    assert len(written) == 1
    assert "key-sample_1" in written[0], f"Expected sample_1 selected, got {written}"
    assert "true0" in written[0]
    assert "pred1" in written[0]


# ---------------------------------------------------------------------------
# AC: all panels logged as a single combined W&B image (not a list)
# ---------------------------------------------------------------------------


def test_wandb_image_logged_when_wandb_active(tmp_path):
    """All selected panels arrive in a single wandb.log call as one combined image."""
    fake_captum, fake_attr = _make_fake_captum_modules()
    trainer = _make_trainer(tmp_path)
    lm = _make_lmodule()
    batch = _make_batch(n=4)
    outputs = _make_outputs(n=4, n_classes=2)

    fake_wandb = MagicMock()
    fake_image_sentinel = MagicMock()
    fake_wandb.Image = MagicMock(return_value=fake_image_sentinel)

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr, "wandb": fake_wandb},
    ):
        import importlib

        import radiologist.core.callbacks.attribution as attr_mod

        importlib.reload(attr_mod)
        AttrCB2 = attr_mod.AttributionCallback
        cb2 = AttrCB2(
            target_layer="0", every_n_val_epochs=1, output_subdir="attributions"
        )
        cb2.on_validation_batch_end(trainer, lm, outputs, batch, batch_idx=0)

    assert fake_wandb.log.call_count == 1
    payload = fake_wandb.log.call_args[0][0]

    gallery_key = "attributions/val"
    assert gallery_key in payload
    assert isinstance(payload[gallery_key], list)
    assert all(img is fake_image_sentinel for img in payload[gallery_key])

    per_sample = {k: v for k, v in payload.items() if k != gallery_key}
    assert all(k.startswith("attributions/val/") for k in per_sample)
    assert all(v is fake_image_sentinel for v in per_sample.values())
