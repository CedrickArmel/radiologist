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

import sys
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture()
def minimal_cfg() -> DictConfig:
    """Minimal DictConfig that disables all heavy operations."""
    return OmegaConf.create(
        {
            "train": False,
            "test": False,
            "seed": 42,
            "ckpt_path": None,
            "optimized_metric": None,
            "extras": {
                "ignore_warnings": False,
                "enforce_tags": False,
                "print_config": False,
            },
        }
    )


def test_train_returns_tuple_of_two_dicts_when_train_and_test_disabled(
    minimal_cfg: DictConfig,
) -> None:
    from radiologist.core.train import train

    result = train(minimal_cfg)

    assert isinstance(result, tuple), "train() must return a tuple"
    assert len(result) == 2, "train() must return a tuple of length 2"
    metrics, obj_dict = result
    assert isinstance(metrics, dict), "first element must be a dict of metrics"
    assert isinstance(obj_dict, dict), "second element must be a dict of objects"


def test_train_reraises_exception_via_task_wrapper(minimal_cfg: DictConfig) -> None:
    """Exceptions raised inside train() propagate after task_wrapper finalises wandb."""
    from radiologist.core.train import train

    bad_cfg = OmegaConf.create(
        {
            "train": True,
            "test": False,
            "seed": 42,
            "ckpt_path": None,
            "optimized_metric": None,
            "extras": {
                "ignore_warnings": False,
                "enforce_tags": False,
                "print_config": False,
            },
            "callbacks": None,
            "loggers": None,
            "trainer": {"_target_": "this.does.not.Exist"},
            "module": {"_target_": "this.does.not.Exist"},
            "datamodule": {"_target_": "this.does.not.Exist"},
        }
    )

    with pytest.raises(Exception):
        train(bad_cfg)


def test_train_public_api_symbols_importable() -> None:
    """All Phase 1-3 public symbols resolve from radiologist.core."""
    from radiologist.core import (  # noqa: F401
        AttributionCallback,
        BestMetricCallback,
        FocalLoss,
        LModule,
        WandbDefineSummaryCallback,
        WebDatasetDataModule,
    )


def test_train_module_importable() -> None:
    """train and main are importable from radiologist.core.train."""
    from radiologist.core.train import main, train  # noqa: F401

    assert callable(train)
    assert callable(main)


def test_configs_train_yaml_exists() -> None:
    """configs/train.yaml exists under radiologist.core package."""
    configs_dir = (
        Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"
    )
    assert (configs_dir / "train.yaml").exists(), "configs/train.yaml must exist"


def test_configs_eval_yaml_exists() -> None:
    """configs/eval.yaml exists under radiologist.core package."""
    configs_dir = (
        Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"
    )
    assert (configs_dir / "eval.yaml").exists(), "configs/eval.yaml must exist"


def test_configs_trainer_yaml_has_use_distributed_sampler_false() -> None:
    """configs/trainer.yaml sets use_distributed_sampler: false (required for WebDataset DDP)."""
    configs_dir = (
        Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"
    )
    cfg = OmegaConf.load(configs_dir / "trainer.yaml")
    assert cfg.trainer.use_distributed_sampler == False  # noqa: E712


def test_configs_callbacks_default_yaml_wires_all_five_callbacks() -> None:
    """callbacks/default.yaml lists all five required callbacks."""
    configs_dir = (
        Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"
    )
    with open(configs_dir / "callbacks" / "default.yaml") as f:
        content = f.read()

    required = [
        "best_metric",
        "wandb_summary",
        "attribution",
        "model_checkpoint",
        "lr_monitor",
    ]
    for name in required:
        assert name in content, f"callbacks/default.yaml must reference '{name}'"


def test_configs_datamodule_num_classes_interpolation_resolves() -> None:
    """${datamodule.num_classes} is a valid interpolation reference in module configs."""
    configs_dir = (
        Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"
    )
    with open(configs_dir / "module" / "resnet50.yaml") as f:
        content = f.read()
    assert "${datamodule.num_classes}" in content


def test_configs_train_yaml_has_correct_optimized_metric() -> None:
    """train.yaml root config has optimized_metric: best_val_score."""
    configs_dir = (
        Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"
    )
    cfg = OmegaConf.load(configs_dir / "train.yaml")
    assert cfg.optimized_metric == "best_val_score"


def test_configs_eval_yaml_has_train_false_test_true() -> None:
    """eval.yaml sets train: false and test: true (eval-only mode)."""
    configs_dir = (
        Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"
    )
    cfg = OmegaConf.load(configs_dir / "eval.yaml")
    assert cfg.train == False  # noqa: E712
    assert cfg.test == True  # noqa: E712
