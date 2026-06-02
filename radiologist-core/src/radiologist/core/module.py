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

from functools import partial
from typing import Any, Dict, List, Optional

import lightning as L
import torch
from torchmetrics import MaxMetric, MeanMetric, Metric


class LModule(L.LightningModule):
    """Lightning module wrapping an arbitrary ``net`` with focal-loss training.

    Args:
        net: backbone network.
        loss: loss function (e.g. FocalLoss).
        metric: ``partial`` factory for the primary metric (e.g. F-beta score).
        optimizer: ``partial`` factory for the optimizer.
        scheduler: optional ``partial`` factory for the LR scheduler.
        trainable_layers: mapping of layer-group names to parameter lists.
        priors: optional class prior probabilities for bias initialisation.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        metric: partial,  # type: ignore[type-arg]
        optimizer: partial,  # type: ignore[type-arg]
        scheduler: Optional[partial] = None,  # type: ignore[type-arg]
        trainable_layers: Optional[Dict[str, List]] = None,
        priors: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss"])

        self.net = net
        self.criterion = loss

        self.val_score: Metric = metric()
        self.test_score: Metric = metric()
        self.train_score: Metric = metric()

        _kw = dict(
            compute_on_cpu=self.val_score.compute_on_cpu,
            sync_on_compute=self.val_score.sync_on_compute,
            process_group=self.val_score.process_group,
        )
        self.val_score_best = MaxMetric(**_kw)
        self.train_loss = MeanMetric(**_kw)
        self.val_loss = MeanMetric(**_kw)
        self.test_loss = MeanMetric(**_kw)

        self.output: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for input tensor ``x``."""
        return self.net(x)

    def configure_optimizers(self) -> Any:
        _opt_factory: partial = self.hparams["optimizer"]  # type: ignore[index]
        _sched_factory: Optional[partial] = self.hparams.get(  # type: ignore[union-attr]
            "scheduler", None
        )
        optimizer = _opt_factory(params=self.parameters())
        if _sched_factory is None:
            return {"optimizer": optimizer}
        scheduler = _sched_factory(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer: Any,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        grad_norm = self._get_grad_norm()
        self.log("grad_norm", grad_norm, on_step=True, prog_bar=False)
        self.clip_gradients(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
        weight_norm = self._get_weight_norm()
        self.log("weight_norm_post_clip", weight_norm, on_step=True, prog_bar=False)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["input"])
        loss = self.criterion(logits, batch["target"])
        preds = logits.argmax(dim=1)
        self.train_loss(loss)
        self.train_score(preds, batch["target"])
        self.log("train_loss", self.train_loss, on_step=True, prog_bar=True)
        self.log("train_score", self.train_score, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["input"])
        loss = self.criterion(logits, batch["target"])
        preds = logits.argmax(dim=1)
        self.val_loss(loss)
        self.val_score(preds, batch["target"])
        self.log("val_loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val_score", self.val_score, on_epoch=True, prog_bar=True)
        return logits

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["input"])
        loss = self.criterion(logits, batch["target"])
        preds = logits.argmax(dim=1)
        self.test_loss(loss)
        self.test_score(preds, batch["target"])
        self.log("test_loss", self.test_loss, on_epoch=True, prog_bar=True)
        self.log("test_score", self.test_score, on_epoch=True, prog_bar=True)
        self.output.append(preds)
        return logits

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        weight_norm = self._get_weight_norm()
        self.log("weight_norm", weight_norm, on_step=True, prog_bar=False)

    def on_validation_epoch_end(self) -> None:
        score = self.val_score.compute()
        self.val_score_best(score)
        self.log("val_score_best", self.val_score_best.compute(), prog_bar=True)

    def _get_grad_norm(self) -> torch.Tensor:
        total_norm = torch.tensor(0.0)
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2) ** 2
        return total_norm**0.5

    def _get_weight_norm(self) -> torch.Tensor:
        total_norm = torch.tensor(0.0)
        for p in self.parameters():
            total_norm += p.data.norm(2) ** 2
        return total_norm**0.5
