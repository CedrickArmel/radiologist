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

import logging
from typing import Any, Callable, Dict, Optional

from omegaconf import DictConfig

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

log = logging.getLogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Apply optional run-time extras controlled by cfg.extras.

    Currently supports: ignoring warnings, enforcing tags, printing config tree.
    """
    if not cfg.get("extras"):
        return

    extras_cfg = cfg.extras

    if extras_cfg.get("ignore_warnings"):
        import warnings

        warnings.filterwarnings("ignore")

    if extras_cfg.get("enforce_tags") and not cfg.get("tags"):
        raise ValueError("You must specify tags for this run.")

    if extras_cfg.get("print_config"):
        log.info("Config:\n%s", cfg)


def task_wrapper(task_func: Callable) -> Callable:
    """Wrap a task function so that wandb is always finalised on exit.

    Re-raises any exception after finalising wandb.
    """

    def wrap(cfg: Any) -> Any:
        try:
            return task_func(cfg)
        except Exception:
            log.exception("Task failed")
            raise
        finally:
            try:
                import wandb as _wandb

                _wandb.finish()
            except ImportError:
                pass

    return wrap


def get_metric_value(
    metric_dict: Dict[str, Any], metric_name: Optional[str]
) -> Optional[float]:
    """Return the scalar value of metric_name from metric_dict.

    Returns 0 when metric_name is falsy.
    Raises KeyError when metric_name is non-falsy but absent from metric_dict.
    """
    if not metric_name:
        return 0
    if metric_name not in metric_dict:
        raise KeyError(
            f"Metric '{metric_name}' not found in metric_dict. "
            f"Available keys: {list(metric_dict.keys())}"
        )
    value = metric_dict[metric_name]
    return value.item() if hasattr(value, "item") else float(value)
