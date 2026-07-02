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

from typing import Any

import lightning as L

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]


class WandbDefineSummaryCallback(L.Callback):
    """Configure the W&B run summary panel for the monitored metric.

    Calls ``wandb.run.define_metric`` for both the raw metric and its
    ``best_`` variant when a W&B run is active.  Silent no-op when wandb
    is absent or no run is active.
    """

    def __init__(self, monitor: str, mode: str = "max") -> None:
        super().__init__()
        self.monitor = monitor
        self.mode = mode

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        run = getattr(wandb, "run", None)
        if run is None:
            return
        run.define_metric(self.monitor, summary=self.mode)
        run.define_metric(f"best_{self.monitor}", summary=self.mode)
