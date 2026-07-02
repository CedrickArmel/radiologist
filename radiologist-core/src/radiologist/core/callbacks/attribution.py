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

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import lightning as L
import torch
import torchvision.utils as vutils  # type: ignore[import-untyped]

try:
    from captum.attr import (  # type: ignore[import-untyped]
        IntegratedGradients,
        LayerGradCam,
    )

    _CAPTUM_AVAILABLE = True
except ImportError:
    IntegratedGradients = None  # type: ignore[assignment,misc]
    LayerGradCam = None  # type: ignore[assignment,misc]
    _CAPTUM_AVAILABLE = False

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


@dataclass
class _Panel:
    """One rendered sample panel and the metadata needed to save and log it.

    Attributes:
        key: dataset key identifying the source sample.
        image: rendered grid ``[3, H', W']`` from ``_build_sample_image``.
        caption: human-readable W&B caption (epoch / key / true / pred).
        filename: deterministic PNG filename (stage, epoch, key, true, pred).
    """

    key: str
    image: torch.Tensor
    caption: str
    filename: str


class AttributionCallback(L.Callback):
    """Writes GradCAM and Integrated-Gradients overlays for val/test batches.

    Selects one sample per true class (preferring misclassified ones), computes
    attributions toward every target class in a single batched call, and logs a
    stacked panel image (one row per class).

    Args:
        target_layer: dot-path resolved lazily against ``pl_module.net``
            (e.g. ``"layer4.1.conv2"``).
        every_n_val_epochs: run on validation epochs where
            ``epoch % every_n_val_epochs == 0``.
        every_n_batches: run on batches where ``batch_idx % every_n_batches == 0``.
        every_n_test_batches: override of ``every_n_batches`` for test stage.
        output_subdir: sub-directory under ``trainer.log_dir`` for PNGs.
        save_to_file: whether to save panel PNGs to disk.
        ig_n_steps: number of IG quadrature steps. ``None`` (default) disables IG
            entirely — IG requires this many forward+backward passes per
            sample×class and dominates wall time.
    """

    def __init__(
        self,
        target_layer: str,
        every_n_val_epochs: int = 1,
        every_n_batches: int = 1,
        every_n_test_batches: Optional[int] = None,
        output_subdir: str = "attributions",
        save_to_file: bool = True,
        ig_n_steps: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.target_layer = target_layer
        self.every_n_val_epochs = every_n_val_epochs
        self._every_n_batches = every_n_batches
        self._every_n_test_batches = every_n_test_batches
        self.output_subdir = output_subdir
        self._save_to_file = save_to_file
        self.ig_n_steps = ig_n_steps

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Union[torch.Tensor, Mapping[str, Any], None],
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if (
            not trainer.is_global_zero
            or batch_idx % self._every_n_batches != 0
            or trainer.current_epoch % self.every_n_val_epochs != 0
        ):
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
        if (
            not trainer.is_global_zero
            or batch_idx % (self._every_n_test_batches or self._every_n_batches) != 0
        ):
            return
        self._run_attribution(trainer, pl_module, outputs, batch, stage="test")

    def _run_attribution(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Union[torch.Tensor, Mapping[str, Any], None],
        batch: Dict[str, Any],
        stage: str,
    ) -> None:
        """Orchestrate: guard → select → batch → attribute → build → save → log."""
        if not _CAPTUM_AVAILABLE:
            return
        log_dir = trainer.log_dir or trainer.default_root_dir
        if log_dir is None:
            return
        if outputs is None or not isinstance(outputs, torch.Tensor):
            return

        inputs = batch["input"]
        targets_true = batch["target"]
        K = outputs.shape[1]
        preds = outputs.argmax(dim=1)

        selected = self._select_indices(targets_true, preds, K)
        if not selected:
            return

        layer = self._resolve_layer(pl_module.net)
        out_dir = self._resolve_out_dir(trainer.log_dir or trainer.default_root_dir)
        inputs_rep, targets_rep = self._prepare_batched_inputs(inputs, selected, K)

        was_training = pl_module.net.training
        pl_module.net.train(mode=False)
        try:
            gc_all, ig_all = self._compute_attributions(
                pl_module.net, layer, inputs_rep, targets_rep
            )
        finally:
            pl_module.net.train(mode=was_training)

        panels = self._build_panels(
            inputs,
            selected,
            batch["key"],
            targets_true,
            preds,
            gc_all,
            ig_all,
            K,
            trainer.current_epoch,
            stage,
        )
        for panel in panels:
            self._save_panel(panel, out_dir)
        self._log_gallery(
            panels, stage, step=trainer.global_step if stage == "val" else None
        )

    def _compute_attributions(
        self,
        net: torch.nn.Module,
        layer: torch.nn.Module,
        inputs_rep: torch.Tensor,
        targets_rep: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Run GradCAM (and IG when enabled) over the batched inputs.

        Exits ``inference_mode`` and re-enables grad so captum can backprop.
        Clones both tensors before attribution — they may be inference tensors
        when called from the test loop, and captum uses targets inside
        ``torch.gather`` on an autograd-tracked output.

        Does not toggle ``net.train`` — the caller owns that lifecycle.

        Args:
            net: the module under attribution.
            layer: target conv layer resolved via ``_resolve_layer``.
            inputs_rep: batched inputs ``[S*K, C, H, W]``.
            targets_rep: per-row target classes ``[S*K]``.

        Returns:
            ``(gc_all, ig_all)``. ``gc_all`` is ``[S*K, 1, h, w]``;
            ``ig_all`` is ``[S*K, C, H, W]`` or ``None`` when IG is disabled.
        """
        gc = LayerGradCam(net, layer)
        ig = IntegratedGradients(net) if self.ig_n_steps is not None else None

        with torch.inference_mode(False), torch.enable_grad():
            inputs_attr = inputs_rep.clone()
            targets_attr = targets_rep.clone()
            gc_all = gc.attribute(inputs_attr, target=targets_attr)  # [S*K, 1, h, w]
            ig_all = (
                ig.attribute(
                    inputs_attr, target=targets_attr, n_steps=self.ig_n_steps
                )  # [S*K, C, H, W]
                if ig is not None
                else None
            )

        return gc_all, ig_all

    def _resolve_layer(self, net: torch.nn.Module) -> torch.nn.Module:
        """Resolve ``self.target_layer`` dot-path against ``net``."""
        return functools.reduce(getattr, self.target_layer.split("."), net)

    def _resolve_out_dir(self, log_dir: str) -> Path:
        p = Path(log_dir) / self.output_subdir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _select_indices(
        self,
        targets_true: torch.Tensor,
        preds: torch.Tensor,
        K: int,
    ) -> List[int]:
        """Return one batch index per class, preferring misclassified samples.

        Args:
            targets_true: ground-truth class indices, shape ``[N]``.
            preds: predicted class indices (``outputs.argmax``), shape ``[N]``.
            K: total number of classes (``outputs.shape[1]``).

        Returns:
            List of at most ``K`` indices into the batch (one per present class).
            Classes absent from this batch are skipped.
        """
        selected: List[int] = []
        for c in range(K):
            in_class = (targets_true == c).nonzero(as_tuple=True)[0]
            if in_class.numel() == 0:
                continue
            # prefer first misclassified; fall back to first correct
            wrong = [idx.item() for idx in in_class if preds[idx] != c]
            selected.append(int(wrong[0] if wrong else in_class[0].item()))
        return selected

    def _prepare_batched_inputs(
        self,
        inputs: torch.Tensor,
        selected: List[int],
        K: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build the captum batch: each selected sample repeated once per class.

        Args:
            inputs: full batch inputs ``[N, C, H, W]``.
            selected: batch indices chosen by ``_select_indices``.
            K: number of target classes.

        Returns:
            ``(inputs_rep, targets_rep)`` where ``inputs_rep`` is
            ``[S*K, C, H, W]`` (each sample repeat-interleaved K times) and
            ``targets_rep`` is ``[S*K]`` cycling ``range(K)`` per sample.
        """
        inputs_rep = inputs[selected].repeat_interleave(K, dim=0)
        targets_rep = torch.arange(K, device=inputs.device).repeat(len(selected))
        return inputs_rep, targets_rep

    def _denorm_input(self, x: torch.Tensor) -> torch.Tensor:
        """Min-max normalise [1,C,H,W] input to [3,H,W] float32 in [0,1]."""
        t = x.detach().cpu().float()[0]  # [C, H, W]
        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)
        lo, hi = t.min(), t.max()
        if (hi - lo).abs() > 1e-8:
            t = (t - lo) / (hi - lo)
        return t

    @torch.no_grad()
    def _attr_to_2d(self, attr: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Reduce [K,C,h,w] attribution to [K,1,H,W] normalised per-map in [0,1].

        Handles both low-res GradCAM maps (upsampled) and full-res IG maps
        (channel-reduced via absolute mean).
        """
        import torch.nn.functional as F

        a = attr.detach().cpu().float()
        a = a.abs().mean(dim=1, keepdim=True) if a.shape[1] > 1 else a.abs()
        if a.shape[2] != H or a.shape[3] != W:
            a = F.interpolate(a, size=(H, W), mode="bilinear", align_corners=False)
        flat = a.view(a.shape[0], -1)
        lo = flat.min(dim=1).values.view(-1, 1, 1, 1)
        hi = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return torch.where((hi - lo).abs() > 1e-8, (a - lo) / (hi - lo), a)  # [K,1,H,W]

    @torch.no_grad()
    def _heatmap(self, attr_2d: torch.Tensor) -> torch.Tensor:
        """Apply jet colormap to [K,1,H,W] attribution, return [K,3,H,W] in [0,1]."""
        import numpy as np

        a = attr_2d[:, 0].numpy()  # [K, H, W]
        try:
            from matplotlib import colormaps  # type: ignore[import-untyped]

            rgb = colormaps["jet"](a)[..., :3].astype("float32")  # [K, H, W, 3]
        except Exception:
            rgb = np.stack([a, np.zeros_like(a), 1.0 - a], axis=-1).astype("float32")
        return torch.from_numpy(rgb).permute(0, 3, 1, 2)  # [K, 3, H, W]

    def _overlay(
        self, base: torch.Tensor, heat: torch.Tensor, alpha: float = 0.5
    ) -> torch.Tensor:
        """Alpha-blend heatmap over base image — both [K,3,H,W] or [3,H,W]."""
        return (alpha * heat + (1.0 - alpha) * base).clamp(0.0, 1.0)

    def _annotate(
        self,
        img: torch.Tensor,
        text: str,
        color: Tuple[int, int, int] = (255, 255, 255),
    ) -> torch.Tensor:
        """Burn ``text`` onto a ``[3, H, W]`` float32 tensor in [0,1].

        Draws a 1-px black shadow then the text in ``color`` for legibility on
        any background. Falls back to the unmodified image when PIL is absent.
        """
        try:
            import numpy as np
            from PIL import Image, ImageDraw  # type: ignore[import-untyped]

            arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
            pil = Image.fromarray(arr)
            draw = ImageDraw.Draw(pil)
            draw.text((2, 2), text, fill=(0, 0, 0))
            draw.text((1, 1), text, fill=color)
            return torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
        except Exception:
            return img

    def _colorbar(self, H: int, W: int = 32) -> torch.Tensor:
        """Vertical jet colorbar strip ``[3, H, W]`` mapping 1→0 (top→bottom).

        Annotates "1.0" at the top and "0.0" at the bottom. Falls back to a
        simple red→blue ramp when matplotlib is absent; skips text when PIL is absent.
        """
        import numpy as np

        vals = np.linspace(1.0, 0.0, H, dtype="float32")
        try:
            from matplotlib import colormaps  # type: ignore[import-untyped]

            rgb = colormaps["jet"](vals)[:, :3]  # [H, 3]
        except Exception:
            rgb = np.stack([vals, np.zeros(H, "float32"), 1.0 - vals], axis=-1)
        strip = np.broadcast_to(rgb[:, None, :], (H, W, 3)).copy()
        tensor = torch.from_numpy(strip).permute(2, 0, 1)  # [3, H, W]
        try:
            from PIL import Image, ImageDraw  # type: ignore[import-untyped]

            arr = (tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            pil = Image.fromarray(arr)
            draw = ImageDraw.Draw(pil)
            for y, label in [(1, "1.0"), (H - 10, "0.0")]:
                draw.text((1, y), label, fill=(0, 0, 0))
                draw.text((0, y - 1), label, fill=(255, 255, 255))
            tensor = torch.from_numpy(np.array(pil)).permute(2, 0, 1).float() / 255.0
        except Exception:
            pass
        return tensor

    @torch.no_grad()
    def _build_sample_image(
        self,
        x: torch.Tensor,
        gc_K: torch.Tensor,
        tc: int,
        pc: int,
        ig_K: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build a single-row panel for one sample.

        Layout: ``[orig | GC-ov×K | (IG-ov×K if enabled)]`` + colorbar strip.
        The original is shown once (annotated with true/pred); each overlay is
        annotated with its target class. Correct predictions use green text,
        wrong predictions use red.

        Args:
            x: original input ``[1, C, H, W]``.
            gc_K: GradCAM attributions for all K target classes ``[K, 1, h, w]``.
            tc: ground-truth class index for this sample.
            pc: predicted class index for this sample.
            ig_K: optional IG attributions ``[K, C, H, W]``; omitted when IG is disabled.

        Returns:
            Grid ``[3, H+pad, N*(W+pad)]`` — single row, N = 1 + K (+ K when IG enabled).
        """
        H, W = x.shape[-2], x.shape[-1]
        K = gc_K.shape[0]
        orig = self._denorm_input(x)  # [3, H, W]

        pred_color = (100, 220, 100) if tc == pc else (220, 100, 100)
        orig_ann = self._annotate(orig, f"T:{tc} P:{pc}", color=pred_color)

        gc_2d = self._attr_to_2d(gc_K, H, W)  # [K, 1, H, W]
        gc_heat = self._heatmap(gc_2d)  # [K, 3, H, W]
        orig_rep = orig.unsqueeze(0).expand(K, -1, -1, -1)  # [K, 3, H, W]
        gc_ov = self._overlay(orig_rep, gc_heat)  # [K, 3, H, W]
        gc_tiles = [self._annotate(gc_ov[k], f"GC→c{k}") for k in range(K)]

        tiles: List[torch.Tensor] = [orig_ann] + gc_tiles

        if ig_K is not None:
            ig_2d = self._attr_to_2d(ig_K, H, W)
            ig_heat = self._heatmap(ig_2d)
            ig_ov = self._overlay(orig_rep, ig_heat)  # [K, 3, H, W]
            tiles += [self._annotate(ig_ov[k], f"IG→c{k}") for k in range(K)]

        grid = vutils.make_grid(torch.stack(tiles), nrow=len(tiles), padding=2)
        legend = self._colorbar(grid.shape[1])  # [3, H_grid, 20]
        return torch.cat([grid, legend], dim=2)

    def _build_panels(
        self,
        inputs: torch.Tensor,
        selected: List[int],
        keys: List[str],
        targets_true: torch.Tensor,
        preds: torch.Tensor,
        gc_all: torch.Tensor,
        ig_all: Optional[torch.Tensor],
        K: int,
        epoch: int,
        stage: str,
    ) -> List[_Panel]:
        """Assemble one ``_Panel`` per selected sample.

        Slices the batched attributions back into per-sample K-blocks, renders
        each panel via ``_build_sample_image``, and computes the deterministic
        filename and caption from epoch / key / true / pred.

        Args:
            inputs: full batch inputs ``[N, C, H, W]``.
            selected: batch indices from ``_select_indices``.
            keys: dataset keys for all batch samples.
            targets_true: ground-truth class indices ``[N]``.
            preds: predicted class indices ``[N]``.
            gc_all: GradCAM attributions ``[S*K, 1, h, w]``.
            ig_all: IG attributions ``[S*K, C, H, W]`` or ``None``.
            K: number of target classes.
            epoch: current training epoch (for filenames / captions).
            stage: ``"val"`` or ``"test"``.

        Returns:
            Panels in ``selected`` order.
        """
        panels: List[_Panel] = []
        for s, idx in enumerate(selected):
            gc_K = gc_all[s * K : (s + 1) * K]
            ig_K = ig_all[s * K : (s + 1) * K] if ig_all is not None else None
            key = keys[idx]
            tc = int(targets_true[idx].item())
            pc = int(preds[idx].item())
            panels.append(
                _Panel(
                    key=key,
                    image=self._build_sample_image(
                        inputs[idx : idx + 1], gc_K, tc, pc, ig_K
                    ),
                    caption=f"ep{epoch:03d} key={key} true={tc} pred={pc}",
                    filename=f"{stage}-ep{epoch:03d}-key-{key}-true{tc}-pred{pc}.png",
                )
            )
        return panels

    def _save_panel(self, panel: _Panel, out_dir: Path) -> None:
        """Write one panel PNG to ``out_dir`` using its deterministic filename.

        No-op when ``self._save_to_file`` is ``False``. Overwrites on rerun.
        """
        if self._save_to_file:
            vutils.save_image(panel.image, str(out_dir / panel.filename))

    def _log_gallery(
        self,
        panels: List[_Panel],
        stage: str,
        step: Optional[int] = None,
    ) -> None:
        """Log panels as a gallery list AND as per-sample keys in one ``wandb.log`` call.

        ``attributions/{stage}`` holds a list of all images → cross-image comparison
        at a given step. ``attributions/{stage}/{key}`` holds each image individually
        → cross-step timeline tracking per sample.

        Validation passes an explicit ``step`` to tie images to the training
        global step. Test omits it — ``global_step`` is frozen at the last
        training value and would trigger W&B's monotonicity warning.
        """
        try:
            import numpy as np

            images = [
                wandb.Image(
                    (p.image.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                    caption=p.caption,
                )
                for p in panels
            ]
            payload: Dict[str, Any] = {f"attributions/{stage}": images}
            payload.update(
                {f"attributions/{stage}/{p.key}": img for p, img in zip(panels, images)}
            )
            if step is not None:
                wandb.log(payload, step=step)
            else:
                wandb.log(payload)
        except Exception:
            pass
