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

import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics import MeanMetric, Metric
from torchmetrics.utilities import dim_zero_cat

from radiologist.utils.ml import initialize_weights


class LModule(L.LightningModule):
    """Lightning module wrapping an arbitrary ``net`` with focal-loss training.

    Args:
        cfg: Configuration containing the following keys. The ``net``,
            ``loss``, ``metric``, ``optimizer``, and ``scheduler`` keys should
            each include a ``_target_`` field so that they can be instantiated
            by Hydra. ``metric``, ``optimizer``, and ``scheduler`` should
            additionally include ``_partial_: true``, since they must be
            partially instantiated.

            net: Backbone network.
            loss: Loss function (e.g. FocalLoss).
            metric: ``partial`` factory for the primary metric (e.g. F-beta score).
            optimizer: ``partial`` factory for the optimizer.
            scheduler: Optional ``partial`` factory for the LR scheduler.
            trainable_layers: Mapping of layer-group names to parameter lists.
            priors: Optional class prior probabilities for bias initialisation.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Instantiate the net, loss, metrics, and optimizer/scheduler from ``cfg``."""
        super().__init__()
        self.save_hyperparameters()

        self.net = instantiate(cfg.get("net"))
        self.criterion = instantiate(cfg.get("loss"))
        self.optimizer = instantiate(cfg.get("optimizer"))
        self.scheduler = instantiate(cfg.get("scheduler"))

        self.hparams["trainable_layers"] = cfg.get("trainable_layers")
        self.hparams["priors"] = cfg.get("priors")

        metric = instantiate(cfg.get("metric"))

        self.val_score: Metric = metric()
        self.test_score: Metric = metric()

        _kw = dict(
            compute_on_cpu=self.val_score.compute_on_cpu,
            sync_on_compute=self.val_score.sync_on_compute,
            process_group=self.val_score.process_group,
        )
        self.train_loss = MeanMetric(**_kw)
        self.val_loss = MeanMetric(**_kw)
        self.test_loss = MeanMetric(**_kw)

        self.output: List[torch.Tensor] = []

    def setup(self, stage: str) -> None:
        """On 'fit': configure trainable params and (optionally) bias priors.

        Transfer-learning (trainable_layers is not None):
            freeze all -> selective unfreeze -> prior bias init if priors set.
        From scratch (trainable_layers is None):
            re-initialise net weights with initialize_weights(net, dist="normal").

        Args:
            stage: the Lightning stage identifier (``"fit"``, ``"validate"``,
                ``"test"``, or ``"predict"``); only ``"fit"`` triggers setup.
        """
        if stage == "fit":
            trainable_layers = self.hparams.get("trainable_layers", None)  # type: ignore[union-attr]
            if trainable_layers is not None:
                self.net.apply(lambda m: self._set_layer_trainable(m, False))
                self._set_trainable()
            else:
                initialize_weights(self.net, dist="normal")

            priors = self.hparams.get("priors", None)  # type: ignore[union-attr]
            if priors is None:
                try:
                    priors = getattr(self.trainer.datamodule, "priors", None)
                except RuntimeError:
                    priors = None

            if priors:
                self._init_last_linear_bias_with_priors(priors=priors)

    def _set_trainable(self) -> None:
        """Unfreeze layers named in self.hparams['trainable_layers'].

        Each key is a dot-path into self.net. Value None unfreezes the whole
        submodule; a list of ints unfreezes only submodule[idx] for each idx.
        """
        trainable_layers: Dict[str, Any] = self.hparams["trainable_layers"]  # type: ignore[index]
        for dot_path, indices in trainable_layers.items():
            submodule = self._resolve_submodule(dot_path)
            if indices is None:
                self._set_layer_trainable(submodule, True)
            else:
                for idx in indices:
                    self._set_layer_trainable(submodule[idx], True)  # type: ignore[index]

    def _resolve_submodule(self, dot_path: str) -> torch.nn.Module:
        """Traverse self.net by dot-path; empty string returns self.net."""
        if dot_path == "":
            return self.net
        module: torch.nn.Module = self.net
        for part in dot_path.split("."):
            if part.isdigit():
                module = module[int(part)]  # type: ignore
            else:
                module = getattr(module, part)
        return module

    def _set_layer_trainable(
        self, layer: torch.nn.Module, trainable: bool = False
    ) -> None:
        """Set requires_grad=trainable on every parameter of layer."""
        for param in layer.parameters():
            param.requires_grad = trainable

    def _init_last_linear_bias_with_priors(self, priors: list[float]) -> None:
        """Set the last nn.Linear bias in self.net to -log(priors).

        Raises:
            ValueError: if len(priors) != out_features of the last Linear bias.
        """
        last_linear: Optional[torch.nn.Linear] = None
        for module in self.net.modules():
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                last_linear = module
        if last_linear is None:
            return
        if len(priors) != last_linear.out_features:
            raise ValueError(
                f"len(priors)={len(priors)} does not match "
                f"out_features={last_linear.out_features}"
            )
        bias_val = -torch.log(torch.tensor(priors, dtype=torch.float32))
        with torch.no_grad():
            last_linear.bias.copy_(bias_val)

    def _shared_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logits, loss, preds) for a batch.

        logits = self(batch['input']); loss = self.criterion(logits, target);
        preds  = logits.argmax(dim=1).
        """
        logits = self(batch["input"])
        loss = self.criterion(logits, batch["target"])
        preds = logits.argmax(dim=1)
        return logits, loss, preds

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for input tensor ``x``."""
        return self.net(x)

    def configure_optimizers(self) -> Any:
        """Build the optimizer and, when configured, the step-wise LR scheduler.

        Returns:
            A dict with key ``"optimizer"``, plus ``"lr_scheduler"`` (with
            ``interval: "step"``) when a scheduler factory is configured.
        """
        _opt_factory: partial = self.optimizer  # type: ignore[index]
        _sched_factory: Optional[partial] = self.scheduler

        optimizer = _opt_factory(params=self.parameters())
        if _sched_factory is None:
            return {"optimizer": optimizer}
        scheduler = _sched_factory(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_before_optimizer_step(self, optimizer: Any) -> None:
        """Log pre-clip gradient L2 norm. Fires every optimizer step."""
        grad_norm = self._get_grad_norm()
        self.log("grad_norm", grad_norm, on_step=True, prog_bar=False)

    def on_save_checkpoint(self, checkpoint):
        """Store the trainer's active precision alongside the checkpoint.

        Args:
            checkpoint: the checkpoint dict Lightning is about to persist.
        """
        checkpoint["precision"] = self.trainer.precision

    def configure_gradient_clipping(
        self,
        optimizer: Any,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Clip gradients, then log post-clip grad norm."""
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        grad_norm_post_clip = self._get_grad_norm()
        self.log(
            "grad_norm_post_clip", grad_norm_post_clip, on_step=True, prog_bar=False
        )

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one training step, logging the running mean training loss.

        Args:
            batch: dict with ``"input"`` and ``"target"`` tensors.
            batch_idx: index of this batch within the epoch.

        Returns:
            The scalar loss tensor for this batch.
        """
        _, loss, _ = self._shared_step(batch)
        self.train_loss(loss)
        self.log("train_loss", self.train_loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one validation step, logging epoch-level loss and score.

        Args:
            batch: dict with ``"input"`` and ``"target"`` tensors.
            batch_idx: index of this batch within the epoch.

        Returns:
            The raw logits tensor for this batch.
        """
        logits, loss, preds = self._shared_step(batch)
        self.val_loss(loss)
        self.val_score(preds, batch["target"])
        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val_score", self.val_score, on_epoch=True, prog_bar=True)
        return logits

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one test step, logging epoch-level loss/score and buffering preds.

        Predictions are appended to ``self.output`` for later persistence in
        ``on_test_epoch_end``.

        Args:
            batch: dict with ``"input"`` and ``"target"`` tensors.
            batch_idx: index of this batch within the epoch.

        Returns:
            The raw logits tensor for this batch.
        """
        logits, loss, preds = self._shared_step(batch)
        self.test_loss(loss)
        self.test_score(preds, batch["target"])
        self.log("test_loss", self.test_loss, on_epoch=True, prog_bar=True)
        self.log("test_score", self.test_score, on_epoch=True, prog_bar=True)
        self.output.append(preds)
        return logits

    def on_test_epoch_end(self) -> None:
        """Concatenate self.output and save to preds-rank{global_rank}.pt.

        No-op when self.trainer.log_dir is None or output buffer is empty.
        Clears the buffer after saving so repeated test() calls don't accumulate.
        """
        if not self.output:
            return
        output = dim_zero_cat(self.output)
        self.output.clear()
        if self.trainer.log_dir:
            path = os.path.join(
                self.trainer.log_dir,
                f"preds-rank{self.trainer.global_rank}.pt",
            )
            torch.save(output, path)

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Log the post-step L2 weight norm. Fires after every training batch.

        Args:
            outputs: the training step's return value (unused).
            batch: the training batch (unused).
            batch_idx: index of the batch just processed (unused).
        """
        weight_norm = self._get_weight_norm()
        self.log("weight_norm", weight_norm, on_step=True, prog_bar=False)

    def _get_grad_norm(self) -> torch.Tensor:
        total_norm = torch.tensor(0.0, device=self.device)
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2) ** 2
        return total_norm**0.5

    def _get_weight_norm(self) -> torch.Tensor:
        total_norm = torch.tensor(0.0, device=self.device)
        for p in self.parameters():
            total_norm += p.data.norm(2) ** 2
        return total_norm**0.5
