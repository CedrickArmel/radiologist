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

from pathlib import Path
from unittest.mock import MagicMock, patch

import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint

from radiologist.core import OnnxExportCallback

INPUT_SHAPE = (1, 3, 8, 8)
CLASSES = ["healthy", "sick"]
CAM_TARGET_LAYER = "2"  # Dropout layer index in the lmodule fixture net


def _real_fit_trainer(tmp_path: Path, callback: OnnxExportCallback):
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(tmp_path / "checkpoints"),
        monitor="val_score",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=str(tmp_path),
        callbacks=[checkpoint_callback, callback],
        logger=False,
    )
    return trainer


# ---------------------------------------------------------------------------
# AC1/AC3: active run + best checkpoint -> det+mcd logged with alias 'best'
# and onnx files exist on disk.
# ---------------------------------------------------------------------------


def test_on_fit_end_logs_best_alias_artifacts_and_writes_onnx_files(
    lmodule, dm, tmp_path
):
    callback = OnnxExportCallback(
        input_shape=INPUT_SHAPE,
        classes=CLASSES,
        cam_target_layer=CAM_TARGET_LAYER,
    )
    trainer = _real_fit_trainer(tmp_path, callback)
    fake_run = MagicMock()
    fake_run.id = "run123"

    with patch.object(wandb, "run", fake_run):
        trainer.fit(lmodule, datamodule=dm)

    assert fake_run.log_artifact.called
    aliases_seen = [
        call.kwargs.get("aliases") for call in fake_run.log_artifact.call_args_list
    ]
    assert ["best"] in aliases_seen

    names_logged = [call.args[0].name for call in fake_run.log_artifact.call_args_list]
    assert any(name.startswith(f"model-{fake_run.id}") for name in names_logged)
    assert any(name.endswith("-mcd") for name in names_logged)


def test_on_fit_end_writes_det_and_mcd_onnx_to_run_output_dir(lmodule, dm, tmp_path):
    callback = OnnxExportCallback(
        input_shape=INPUT_SHAPE,
        classes=CLASSES,
        cam_target_layer=CAM_TARGET_LAYER,
    )
    trainer = _real_fit_trainer(tmp_path, callback)
    fake_run = MagicMock()
    fake_run.id = "run456"

    with patch.object(wandb, "run", fake_run):
        trainer.fit(lmodule, datamodule=dm)

    out_dir = Path(trainer.log_dir or trainer.default_root_dir)
    onnx_files = list(out_dir.rglob("*.onnx"))
    det_files = [f for f in onnx_files if not f.name.endswith("-mcd.onnx")]
    mcd_files = [f for f in onnx_files if f.name.endswith("-mcd.onnx")]
    assert det_files, "deterministic .onnx must exist on disk"
    assert mcd_files, "mc-dropout .onnx must exist on disk"


# ---------------------------------------------------------------------------
# AC2: last checkpoint also present -> model-{run_id} logged with alias 'last'
# ---------------------------------------------------------------------------


def test_on_fit_end_logs_last_alias_when_last_checkpoint_exists(lmodule, dm, tmp_path):
    callback = OnnxExportCallback(
        input_shape=INPUT_SHAPE,
        classes=CLASSES,
        cam_target_layer=CAM_TARGET_LAYER,
    )
    trainer = _real_fit_trainer(tmp_path, callback)
    fake_run = MagicMock()
    fake_run.id = "run789"

    with patch.object(wandb, "run", fake_run):
        trainer.fit(lmodule, datamodule=dm)

    assert trainer.checkpoint_callback.last_model_path
    aliases_seen = [
        call.kwargs.get("aliases") for call in fake_run.log_artifact.call_args_list
    ]
    assert ["last"] in aliases_seen


# ---------------------------------------------------------------------------
# Silent no-op: no active W&B run
# ---------------------------------------------------------------------------


def test_on_fit_end_noop_when_no_active_wandb_run(lmodule, dm, tmp_path):
    callback = OnnxExportCallback(
        input_shape=INPUT_SHAPE,
        classes=CLASSES,
        cam_target_layer=CAM_TARGET_LAYER,
    )
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback],
        logger=False,
    )

    with patch.object(wandb, "run", None):
        trainer.fit(lmodule, datamodule=dm)

    out_dir = Path(trainer.log_dir or trainer.default_root_dir)
    assert list(out_dir.rglob("*.onnx")) == []


# ---------------------------------------------------------------------------
# Silent no-op: no best checkpoint available (no ModelCheckpoint configured)
# ---------------------------------------------------------------------------


def test_on_fit_end_noop_when_no_best_checkpoint_available(lmodule, dm, tmp_path):
    callback = OnnxExportCallback(
        input_shape=INPUT_SHAPE,
        classes=CLASSES,
        cam_target_layer=CAM_TARGET_LAYER,
    )
    trainer = L.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        callbacks=[callback],
        logger=False,
    )
    fake_run = MagicMock()

    with patch.object(wandb, "run", fake_run):
        trainer.fit(lmodule, datamodule=dm)

    assert not fake_run.log_artifact.called


# ---------------------------------------------------------------------------
# This slice never links to a collection / applies production|staging alias
# ---------------------------------------------------------------------------


def test_on_fit_end_never_links_artifacts_to_a_collection(lmodule, dm, tmp_path):
    callback = OnnxExportCallback(
        input_shape=INPUT_SHAPE,
        classes=CLASSES,
        cam_target_layer=CAM_TARGET_LAYER,
    )
    trainer = _real_fit_trainer(tmp_path, callback)
    fake_run = MagicMock()
    fake_run.id = "runlink"

    with patch.object(wandb, "run", fake_run):
        trainer.fit(lmodule, datamodule=dm)

    assert fake_run.log_artifact.called
    assert not fake_run.link_artifact.called
