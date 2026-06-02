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

import importlib.util
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf


def _load_rich_utils():
    src = Path(__file__).parent.parent / "src"
    spec = importlib.util.spec_from_file_location(
        "radiologist.utils.ml.rich_utils",
        src / "radiologist" / "utils" / "ml" / "rich_utils.py",
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["radiologist.utils.ml.rich_utils"] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


rich_utils = _load_rich_utils()
print_config_tree = rich_utils.print_config_tree
enforce_tags = rich_utils.enforce_tags


# ---------------------------------------------------------------------------
# print_config_tree
# ---------------------------------------------------------------------------


def test_print_config_tree_runs_without_error_on_minimal_cfg() -> None:
    cfg = OmegaConf.create({"model": {"name": "vgg11"}, "trainer": {"max_epochs": 5}})
    print_config_tree(cfg)


def test_print_config_tree_runs_without_error_when_resolve_is_true() -> None:
    cfg = OmegaConf.create({"lr": 0.001, "trainer": {"lr": "${lr}"}})
    print_config_tree(cfg, resolve=True)


def test_print_config_tree_respects_print_order() -> None:
    cfg = OmegaConf.create({"z_last": 1, "a_first": 2})
    print_config_tree(cfg, print_order=["a_first"])


def test_print_config_tree_skips_save_when_output_dir_absent() -> None:
    cfg = OmegaConf.create({"model": {"name": "vgg11"}})
    print_config_tree(cfg, save_to_file=True)


# ---------------------------------------------------------------------------
# enforce_tags
# ---------------------------------------------------------------------------


def test_enforce_tags_is_noop_when_tags_present() -> None:
    cfg = OmegaConf.create(
        {"tags": ["experiment_1"], "hydra": {"runtime": {"choices": {}}}}
    )
    enforce_tags(cfg)


def test_enforce_tags_raises_value_error_during_multirun_with_no_tags() -> None:
    cfg = OmegaConf.create(
        {
            "tags": [],
            "hydra": {"runtime": {"choices": {"hydra/sweeper": "multirun"}}},
        }
    )
    with pytest.raises(ValueError, match="[Tt]ag"):
        enforce_tags(cfg)


def test_enforce_tags_raises_value_error_during_multirun_with_none_tags() -> None:
    cfg = OmegaConf.create(
        {
            "tags": None,
            "hydra": {"runtime": {"choices": {"hydra/sweeper": "multirun"}}},
        }
    )
    with pytest.raises(ValueError, match="[Tt]ag"):
        enforce_tags(cfg)


def test_enforce_tags_skips_save_when_output_dir_absent() -> None:
    cfg = OmegaConf.create({"tags": ["exp"], "hydra": {"runtime": {"choices": {}}}})
    enforce_tags(cfg, save_to_file=True)


# ---------------------------------------------------------------------------
# module API
# ---------------------------------------------------------------------------


def test_module_exposes_print_config_tree_and_enforce_tags() -> None:
    assert callable(rich_utils.print_config_tree)
    assert callable(rich_utils.enforce_tags)
