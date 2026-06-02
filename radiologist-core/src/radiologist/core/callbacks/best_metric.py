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

from typing import Any

import lightning as L
import torch


class BestMetricCallback(L.Callback):
    """Track the best value of a monitored metric across validation epochs.

    Logs ``best_{monitor}`` to every trainer logger so Optuna's
    ``get_metric_value`` can always retrieve the run's best result.
    """

    def __init__(self, monitor: str, mode: str = "max") -> None:
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self._best: Any = None

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        value = current.item() if isinstance(current, torch.Tensor) else float(current)

        if self._best is None:
            improved = True
        elif self.mode == "max":
            improved = value > self._best
        else:
            improved = value < self._best

        if improved:
            self._best = value

        key = f"best_{self.monitor}"
        trainer.callback_metrics[key] = self._best

        for logger in trainer.loggers:
            logger.log_metrics({key: self._best})
