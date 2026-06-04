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

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers — tiny nn.Sequential that stands in as pl_module.net
# ---------------------------------------------------------------------------


def _make_net() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 2),
    )


def _make_pl_module(net: nn.Module) -> MagicMock:
    pl = MagicMock()
    pl.net = net
    pl.training = True
    return pl


def _make_trainer(tmp_path: Path, epoch: int = 0) -> MagicMock:
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.log_dir = str(tmp_path)
    return trainer


def _make_batch(n: int = 4) -> dict:
    return {
        "input": torch.randn(n, 1, 8, 8),
        "target": torch.zeros(n, dtype=torch.long),
        "key": [f"sample_{i}" for i in range(n)],
    }


def _make_outputs(n: int = 4) -> torch.Tensor:
    return torch.randn(n, 2)


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
    # Patch captum so the callback believes captum IS available
    fake_captum = types.ModuleType("captum")
    fake_attr = types.ModuleType("captum.attr")
    fake_captum.attr = fake_attr

    mock_gradcam_cls = MagicMock()
    mock_ig_cls = MagicMock()
    fake_attr.LayerGradCam = mock_gradcam_cls
    fake_attr.IntegratedGradients = mock_ig_cls

    import sys

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr},
    ):
        # Re-import to pick up the patched modules
        import importlib

        import radiologist.core.callbacks.attribution as attr_mod

        importlib.reload(attr_mod)
        AttrCB = attr_mod.AttributionCallback

        cb = AttrCB(target_layer="bad.path.does.not.exist", every_n_val_epochs=1)
        trainer = _make_trainer(tmp_path, epoch=0)
        net = _make_net()
        pl = _make_pl_module(net)
        batch = _make_batch()
        outputs = _make_outputs()

        with pytest.raises(AttributeError):
            cb.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=0)


# ---------------------------------------------------------------------------
# AC: batch_idx != 0 → no-op (no files written)
# ---------------------------------------------------------------------------


def test_validation_batch_idx_nonzero_writes_no_files(tmp_path):
    from radiologist.core import AttributionCallback

    cb = AttributionCallback(target_layer="0", every_n_val_epochs=1)
    trainer = _make_trainer(tmp_path, epoch=0)
    pl = _make_pl_module(_make_net())
    batch = _make_batch()
    outputs = _make_outputs()

    cb.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=1)

    written = list(tmp_path.rglob("*.png"))
    assert written == []


# ---------------------------------------------------------------------------
# AC: epoch % every_n_val_epochs != 0 → no-op even for batch_idx=0
# ---------------------------------------------------------------------------


def test_validation_skipped_on_wrong_epoch(tmp_path):
    from radiologist.core import AttributionCallback

    cb = AttributionCallback(target_layer="0", every_n_val_epochs=3)
    trainer = _make_trainer(tmp_path, epoch=1)
    pl = _make_pl_module(_make_net())
    batch = _make_batch()
    outputs = _make_outputs()

    cb.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=0)

    written = list(tmp_path.rglob("*.png"))
    assert written == []


# ---------------------------------------------------------------------------
# AC: captum absent → silent no-op, no files, no exception
# ---------------------------------------------------------------------------


def _reload_attribution_without_captum():
    """Reload attribution module with captum removed from sys.modules."""
    import importlib
    import sys

    import radiologist.core.callbacks.attribution as attr_mod

    with patch.dict(sys.modules, {"captum": None, "captum.attr": None}):
        importlib.reload(attr_mod)
        return attr_mod.AttributionCallback


def test_no_png_written_when_captum_absent(tmp_path):
    AttrCB = _reload_attribution_without_captum()

    cb = AttrCB(target_layer="0", every_n_val_epochs=1)
    trainer = _make_trainer(tmp_path, epoch=0)
    pl = _make_pl_module(_make_net())
    batch = _make_batch()
    outputs = _make_outputs()

    cb.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=0)

    written = list(tmp_path.rglob("*.png"))
    assert written == []


def test_no_exception_when_captum_absent_test_batch(tmp_path):
    AttrCB = _reload_attribution_without_captum()

    cb = AttrCB(target_layer="0", n_test_batches=2)
    trainer = _make_trainer(tmp_path, epoch=0)
    pl = _make_pl_module(_make_net())
    batch = _make_batch()
    outputs = _make_outputs()

    cb.on_test_batch_end(trainer, pl, outputs, batch, batch_idx=0)

    written = list(tmp_path.rglob("*.png"))
    assert written == []


# ---------------------------------------------------------------------------
# AC: captum present → PNGs written with deterministic names
# ---------------------------------------------------------------------------


def _make_fake_captum_modules():
    """Return (fake_captum, fake_attr) with working stubs."""
    fake_captum = types.ModuleType("captum")
    fake_attr = types.ModuleType("captum.attr")
    fake_captum.attr = fake_attr

    def _fake_gradcam_attr(*args, **kwargs):
        # returns shape [N, 1, H, W] as LayerGradCam does
        n = 1
        return torch.ones(n, 1, 8, 8)

    def _fake_ig_attr(*args, **kwargs):
        return torch.ones(1, 1, 8, 8)

    gradcam_instance = MagicMock()
    gradcam_instance.attribute = MagicMock(side_effect=_fake_gradcam_attr)
    ig_instance = MagicMock()
    ig_instance.attribute = MagicMock(side_effect=_fake_ig_attr)

    fake_attr.LayerGradCam = MagicMock(return_value=gradcam_instance)
    fake_attr.IntegratedGradients = MagicMock(return_value=ig_instance)
    return fake_captum, fake_attr


def _reload_attribution_with_captum(fake_captum, fake_attr):
    import importlib
    import sys

    import radiologist.core.callbacks.attribution as attr_mod

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr},
    ):
        importlib.reload(attr_mod)
        return attr_mod.AttributionCallback


def test_png_files_written_with_captum_on_validation(tmp_path):
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    n_samples = 2
    cb = AttrCB(
        target_layer="0",
        every_n_val_epochs=1,
        n_samples_per_batch=n_samples,
        output_subdir="attributions",
    )
    trainer = _make_trainer(tmp_path, epoch=0)
    net = _make_net()
    pl = _make_pl_module(net)
    batch = _make_batch(n=4)
    outputs = _make_outputs(n=4)

    import sys

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr},
    ):
        cb.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=0)

    out_dir = tmp_path / "attributions"
    written = sorted(p.name for p in out_dir.glob("*.png"))

    for i in range(n_samples):
        key = f"sample_{i}"
        assert f"gradcam-val-ep000-{key}.png" in written
        assert f"ig-val-ep000-{key}.png" in written


def test_png_filenames_are_deterministic_overwrite_on_rerun(tmp_path):
    """Second call with same epoch/batch must produce the same filenames."""
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(
        target_layer="0",
        every_n_val_epochs=1,
        n_samples_per_batch=1,
        output_subdir="attributions",
    )
    trainer = _make_trainer(tmp_path, epoch=2)
    net = _make_net()
    pl = _make_pl_module(net)
    batch = _make_batch(n=2)
    outputs = _make_outputs(n=2)

    import sys

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr},
    ):
        cb.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=0)
        first_mtimes = {
            p: p.stat().st_mtime for p in (tmp_path / "attributions").glob("*.png")
        }
        cb.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=0)
        second_files = set(p.name for p in (tmp_path / "attributions").glob("*.png"))

    first_files = set(p.name for p in first_mtimes)
    assert first_files == second_files, "second run produced different filenames"


def test_test_batch_end_writes_pngs_for_first_n_batches(tmp_path):
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    n_samples = 1
    cb = AttrCB(
        target_layer="0",
        n_test_batches=2,
        n_samples_per_batch=n_samples,
        output_subdir="attributions",
    )
    trainer = _make_trainer(tmp_path, epoch=5)
    net = _make_net()
    pl = _make_pl_module(net)
    batch = _make_batch(n=2)
    outputs = _make_outputs(n=2)

    import sys

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr},
    ):
        cb.on_test_batch_end(trainer, pl, outputs, batch, batch_idx=0)
        cb.on_test_batch_end(trainer, pl, outputs, batch, batch_idx=1)
        cb.on_test_batch_end(trainer, pl, outputs, batch, batch_idx=2)

    out_dir = tmp_path / "attributions"
    written = sorted(p.name for p in out_dir.glob("*.png"))

    key = "sample_0"
    assert f"gradcam-test-ep005-{key}.png" in written
    assert f"ig-test-ep005-{key}.png" in written

    # batch_idx=2 must NOT produce files (beyond n_test_batches=2)
    for fname in written:
        assert "-test-ep005-" in fname


def test_wandb_image_logged_when_wandb_active(tmp_path):
    fake_captum, fake_attr = _make_fake_captum_modules()
    trainer = _make_trainer(tmp_path, epoch=0)
    net = _make_net()
    pl = _make_pl_module(net)
    batch = _make_batch(n=2)
    outputs = _make_outputs(n=2)

    fake_wandb = MagicMock()
    fake_image = MagicMock()
    fake_wandb.Image = MagicMock(return_value=fake_image)

    import sys

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr, "wandb": fake_wandb},
    ):
        import importlib

        import radiologist.core.callbacks.attribution as attr_mod

        importlib.reload(attr_mod)
        AttrCB2 = attr_mod.AttributionCallback
        cb2 = AttrCB2(
            target_layer="0",
            every_n_val_epochs=1,
            n_samples_per_batch=1,
            output_subdir="attributions",
        )
        cb2.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=0)

    assert fake_wandb.log.called or fake_wandb.Image.called


# ---------------------------------------------------------------------------
# AC: non-global-zero rank → no attribution, no files written
# ---------------------------------------------------------------------------


def _make_trainer_non_zero_rank(tmp_path: Path, epoch: int = 0) -> MagicMock:
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.log_dir = str(tmp_path)
    trainer.is_global_zero = False
    return trainer


def test_validation_batch_end_skips_attribution_on_non_zero_rank(tmp_path):
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", every_n_val_epochs=1, n_samples_per_batch=2)
    trainer = _make_trainer_non_zero_rank(tmp_path, epoch=0)
    pl = _make_pl_module(_make_net())
    batch = _make_batch(n=4)
    outputs = _make_outputs(n=4)

    import sys

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr},
    ):
        cb.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=0)

    written = list(tmp_path.rglob("*.png"))
    assert written == []


def test_test_batch_end_skips_attribution_on_non_zero_rank(tmp_path):
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", n_test_batches=2, n_samples_per_batch=2)
    trainer = _make_trainer_non_zero_rank(tmp_path, epoch=0)
    pl = _make_pl_module(_make_net())
    batch = _make_batch(n=4)
    outputs = _make_outputs(n=4)

    import sys

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr},
    ):
        cb.on_test_batch_end(trainer, pl, outputs, batch, batch_idx=0)

    written = list(tmp_path.rglob("*.png"))
    assert written == []


# ---------------------------------------------------------------------------
# AC: log_dir=None, default_root_dir=None → returns without raising TypeError
# ---------------------------------------------------------------------------


def _make_trainer_null_log_dir(epoch: int = 0) -> MagicMock:
    trainer = MagicMock()
    trainer.current_epoch = epoch
    trainer.log_dir = None
    trainer.default_root_dir = None
    trainer.is_global_zero = True
    return trainer


def test_validation_batch_end_no_error_when_log_dir_is_none():
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", every_n_val_epochs=1, n_samples_per_batch=1)
    trainer = _make_trainer_null_log_dir(epoch=0)
    pl = _make_pl_module(_make_net())
    batch = _make_batch(n=2)
    outputs = _make_outputs(n=2)

    import sys

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr},
    ):
        cb.on_validation_batch_end(trainer, pl, outputs, batch, batch_idx=0)


def test_test_batch_end_no_error_when_log_dir_is_none():
    fake_captum, fake_attr = _make_fake_captum_modules()
    AttrCB = _reload_attribution_with_captum(fake_captum, fake_attr)

    cb = AttrCB(target_layer="0", n_test_batches=2, n_samples_per_batch=1)
    trainer = _make_trainer_null_log_dir(epoch=0)
    pl = _make_pl_module(_make_net())
    batch = _make_batch(n=2)
    outputs = _make_outputs(n=2)

    import sys

    with patch.dict(
        sys.modules,
        {"captum": fake_captum, "captum.attr": fake_attr},
    ):
        cb.on_test_batch_end(trainer, pl, outputs, batch, batch_idx=0)
