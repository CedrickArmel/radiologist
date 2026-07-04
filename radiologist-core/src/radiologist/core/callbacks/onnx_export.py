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

"""Callback that exports the best checkpoint to ONNX and logs it to W&B."""

from typing import Any, List, Tuple

import lightning as L

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

from radiologist.core.registry import export_onnx
from radiologist.registry import WandbRegistry


class OnnxExportCallback(L.Callback):
    """Opt-in end-of-fit ONNX export that logs the model to the active W&B run.

    Silent no-op when wandb has no active run or no best checkpoint exists.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        classes: List[str],
        cam_target_layer: str,
        opset: int = 18,
    ) -> None:
        """Initialize the callback with the export shape, classes, and ONNX opset.

        Args:
            input_shape: shape of the dummy input tensor used to trace the
                model for ONNX export, e.g. ``(1, 3, 224, 224)``.
            classes: ordered class names, embedded in the exported artifacts.
            cam_target_layer: dot-path to the conv layer used for the
                deterministic export's GradCAM forward hook.
            opset: ONNX opset version to export with.
        """
        super().__init__()
        self.input_shape = input_shape
        self.classes = classes
        self.cam_target_layer = cam_target_layer
        self.opset = opset
        self._registry = WandbRegistry()

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        """Export the best checkpoint to ONNX and log it to the active W&B run.

        No-op when no W&B run is active or when the trainer has no best
        checkpoint recorded.

        Args:
            trainer: the active ``lightning.Trainer``.
            pl_module: the ``LightningModule`` that was trained.
        """
        run = getattr(wandb, "run", None)
        if run is None:
            return

        checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
        best_ckpt = getattr(checkpoint_callback, "best_model_path", "")
        if not best_ckpt:
            return
        last_ckpt = getattr(checkpoint_callback, "last_model_path", "") or None

        out_dir = trainer.log_dir or trainer.default_root_dir
        result = export_onnx(
            ckpt_path=best_ckpt,
            run_id=run.id,
            input_shape=self.input_shape,
            classes=self.classes,
            cam_target_layer=self.cam_target_layer,
            out_dir=out_dir,
            opset=self.opset,
        )
        self._registry.log_model_artifacts(result, run, best_ckpt, last_ckpt)
