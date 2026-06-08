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

_CONFIGS_DIR = Path(__file__).parent.parent / "src" / "radiologist" / "core" / "configs"


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


# ---------------------------------------------------------------------------
# AC: config files exist and carry required wiring contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel_path",
    [
        "train.yaml",
        "eval.yaml",
    ],
)
def test_config_yaml_exists(rel_path: str) -> None:
    assert (_CONFIGS_DIR / rel_path).exists(), f"configs/{rel_path} must exist"


@pytest.mark.parametrize(
    "check_id,description,check",
    [
        (
            "train_optimized_metric",
            "train.yaml has optimized_metric: best_val_score",
            lambda: OmegaConf.load(_CONFIGS_DIR / "train.yaml").optimized_metric
            == "best_val_score",
        ),
        (
            "eval_train_false_test_true",
            "eval.yaml sets train=false, test=true",
            lambda: (
                OmegaConf.load(_CONFIGS_DIR / "eval.yaml").train == False  # noqa: E712
                and OmegaConf.load(_CONFIGS_DIR / "eval.yaml").test
                == True  # noqa: E712
            ),
        ),
        (
            "trainer_no_distributed_sampler",
            "trainer.yaml sets use_distributed_sampler: false",
            lambda: OmegaConf.load(
                _CONFIGS_DIR / "trainer.yaml"
            ).trainer.use_distributed_sampler
            == False,  # noqa: E712
        ),
        (
            "callbacks_default_wires_all_five",
            "callbacks/default.yaml lists all five required callbacks",
            lambda: all(
                name in (_CONFIGS_DIR / "callbacks" / "default.yaml").read_text()
                for name in [
                    "best_metric",
                    "wandb_summary",
                    "attribution",
                    "model_checkpoint",
                    "lr_monitor",
                ]
            ),
        ),
        (
            "module_interpolation",
            "module/resnet50.yaml references ${datamodule.num_classes}",
            lambda: "${datamodule.num_classes}"
            in (_CONFIGS_DIR / "module" / "resnet50.yaml").read_text(),
        ),
    ],
    ids=lambda x: x if isinstance(x, str) else "",
)
def test_config_contract(check_id: str, description: str, check) -> None:
    assert check(), description
