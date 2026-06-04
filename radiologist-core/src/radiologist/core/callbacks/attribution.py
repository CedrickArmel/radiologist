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

import functools
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import lightning as L
import torch

try:
    from captum.attr import IntegratedGradients, LayerGradCam

    _CAPTUM_AVAILABLE = True
except ImportError:
    _CAPTUM_AVAILABLE = False

try:
    import wandb as _wandb

    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


class AttributionCallback(L.Callback):
    """Writes GradCAM and Integrated-Gradients overlays for val/test batches.

    Args:
        target_layer: dot-path resolved lazily against ``pl_module.net``
            (e.g. ``"features.28"``).
        every_n_val_epochs: run on validation epochs where
            ``epoch % every_n_val_epochs == 0``.
        n_test_batches: number of test batches to process.
        n_samples_per_batch: number of samples per batch to attribute.
        output_subdir: sub-directory under ``trainer.log_dir`` for PNGs.
    """

    def __init__(
        self,
        target_layer: str,
        every_n_val_epochs: int = 1,
        n_test_batches: int = 1,
        n_samples_per_batch: int = 4,
        output_subdir: str = "attributions",
    ) -> None:
        super().__init__()
        self.target_layer = target_layer
        self.every_n_val_epochs = every_n_val_epochs
        self.n_test_batches = n_test_batches
        self.n_samples_per_batch = n_samples_per_batch
        self.output_subdir = output_subdir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_layer(self, net: torch.nn.Module) -> torch.nn.Module:
        """Resolve ``self.target_layer`` dot-path against ``net``."""
        return functools.reduce(getattr, self.target_layer.split("."), net)

    def _out_dir(self, log_dir: str) -> Path:
        p = Path(log_dir) / self.output_subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _run_attribution(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Union[torch.Tensor, Mapping[str, Any], None],
        batch: Dict[str, Any],
        stage: str,
    ) -> None:
        """Core attribution logic — only called when captum is available."""
        log_dir = trainer.log_dir or trainer.default_root_dir
        if log_dir is None:
            return
        layer = self._resolve_layer(pl_module.net)
        out_dir = self._out_dir(log_dir)
        epoch = trainer.current_epoch

        inputs = batch["input"]
        keys = batch["key"]
        if outputs is None or not isinstance(outputs, torch.Tensor):
            return
        targets = outputs.argmax(dim=1)

        n = min(self.n_samples_per_batch, inputs.shape[0])

        was_training = pl_module.net.training
        pl_module.net.train(mode=False)

        try:
            gc = LayerGradCam(pl_module.net, layer)
            ig = IntegratedGradients(pl_module.net)

            for i in range(n):
                x = inputs[i : i + 1]
                t = int(targets[i].item())
                key = keys[i]

                gc_attr = gc.attribute(x, target=t)
                ig_attr = ig.attribute(x, target=t)

                self._save_and_log(
                    gc_attr,
                    out_dir / f"gradcam-{stage}-ep{epoch:03d}-{key}.png",
                    f"gradcam/{stage}/ep{epoch:03d}/{key}",
                )
                self._save_and_log(
                    ig_attr,
                    out_dir / f"ig-{stage}-ep{epoch:03d}-{key}.png",
                    f"ig/{stage}/ep{epoch:03d}/{key}",
                )
        finally:
            pl_module.net.train(mode=was_training)

    def _save_and_log(
        self,
        attr: torch.Tensor,
        path: Path,
        log_key: str,
    ) -> None:
        """Save attribution tensor as PNG and optionally log to W&B."""
        import torchvision.utils as vutils  # type: ignore[import-untyped]

        # attr may be [1, 1, H, W] or [1, C, H, W]; normalise to [0, 1]
        a = attr.detach().cpu().float()
        a_min, a_max = a.min(), a.max()
        if (a_max - a_min).abs() > 1e-8:
            a = (a - a_min) / (a_max - a_min)
        vutils.save_image(a, str(path))

        if _WANDB_AVAILABLE:
            try:
                _wandb.log({log_key: _wandb.Image(str(path))})
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Union[torch.Tensor, Mapping[str, Any], None],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.is_global_zero:
            return
        if not _CAPTUM_AVAILABLE:
            return
        if batch_idx != 0:
            return
        if trainer.current_epoch % self.every_n_val_epochs != 0:
            return
        self._run_attribution(trainer, pl_module, outputs, batch, stage="val")

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Union[torch.Tensor, Mapping[str, Any], None],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not trainer.is_global_zero:
            return
        if not _CAPTUM_AVAILABLE:
            return
        if batch_idx >= self.n_test_batches:
            return
        self._run_attribution(trainer, pl_module, outputs, batch, stage="test")
