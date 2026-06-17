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

from typing import Any, Dict, Optional, Tuple

import hydra
from omegaconf import DictConfig

from radiologist.utils.ml import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    set_seed,
    task_wrapper,
)

try:
    from hydra.utils import instantiate
    from lightning.pytorch import Trainer
except ImportError:
    Trainer = object  # type: ignore[assignment,misc]
    instantiate = None  # type: ignore[assignment]

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Instantiate all components from cfg and run fit + optional test.

    Args:
        cfg: Hydra DictConfig with keys train, test, seed, ckpt_path, callbacks,
            loggers, trainer, module, datamodule.

    Returns:
        Tuple of (metric_dict, object_dict).
    """

    if cfg.get("seed"):
        set_seed(cfg.seed)

    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("loggers"))

    object_dict: Dict[str, Any] = {}
    metric_dict: Dict[str, Any] = {}

    if cfg.get("train") or cfg.get("test"):
        datamodule = instantiate(cfg.datamodule)
        module = instantiate(cfg.module)

        if cfg.get("ckpt_path") and module.precision:
            cfg.trainer.precision = module.precision

        trainer: Trainer = instantiate(
            cfg.trainer,
            callbacks=callbacks,
            logger=loggers,
        )

        object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "module": module,
            "trainer": trainer,
        }

        log.info(f"Output dir: {cfg.paths.output_dir}")

        if cfg.get("train"):
            log.info("Starting fit stage...")
            if cfg.get("ckpt_path"):
                log.debug(
                    f"Resuming training from checkpoint {cfg.get('ckpt_path')}..."
                )
            trainer.fit(
                model=module, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
            )

        # Snapshot fit-phase metrics (e.g. best_val_score) before test overwrites them.
        metric_dict = {**trainer.callback_metrics}

        if cfg.get("test"):
            log.info("Starting test stage...")
            ckpt_path = (
                trainer.checkpoint_callback.best_model_path  # type: ignore[union-attr]
                if cfg.get("train")
                else cfg.get("ckpt_path")
            )

            log.debug(
                "Re-using best model from training..."
                if cfg.get("train")
                else f"Re-using checkpoint {ckpt_path}"
            )

            trainer.test(model=module, datamodule=datamodule, ckpt_path=ckpt_path)
            metric_dict.update(trainer.callback_metrics)

        log_hyperparameters(object_dict)

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    """Hydra entry point: apply extras, run train, return optimized metric."""
    extras(cfg)
    metrics, _ = train(cfg)
    return get_metric_value(metrics, cfg.get("optimized_metric"))


if __name__ == "__main__":
    main()
