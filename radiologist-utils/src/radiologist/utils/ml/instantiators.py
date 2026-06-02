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
from typing import List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import SequentialLR

try:
    from lightning.pytorch import Callback
    from lightning.pytorch.loggers import Logger
except ImportError:
    Callback = object  # type: ignore[assignment,misc]
    Logger = object  # type: ignore[assignment,misc]


def instantiate_callbacks(
    callbacks_cfg: Optional[DictConfig],
) -> Optional[List[Callback]]:
    """Instantiate Lightning callbacks from a DictConfig section.

    Each top-level key must carry a ``_target_`` field.
    Returns None when cfg is empty or falsy; raises TypeError for non-DictConfig.
    """
    if callbacks_cfg is None:
        return None
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError(
            f"callbacks_cfg must be a DictConfig, got {type(callbacks_cfg).__name__}"
        )
    if not callbacks_cfg:
        return None
    return [instantiate(v) for v in callbacks_cfg.values()]


def instantiate_loggers(
    logger_cfg: Optional[DictConfig],
) -> Optional[List[Logger]]:
    """Instantiate Lightning loggers from a DictConfig section.

    Returns None when cfg is empty or falsy; raises TypeError for non-DictConfig.
    """
    if logger_cfg is None:
        return None
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError(
            f"logger_cfg must be a DictConfig, got {type(logger_cfg).__name__}"
        )
    if not logger_cfg:
        return None
    return [instantiate(v) for v in logger_cfg.values()]


def sequential_scheduler(
    optimizer: Optimizer,
    schedulers: List[partial],  # type: ignore[type-arg]
    milestones: List[int],
) -> SequentialLR:
    """Build a SequentialLR from partial scheduler factories and milestone epochs."""
    materialized = [s(optimizer) for s in schedulers]
    return SequentialLR(optimizer, schedulers=materialized, milestones=milestones)
