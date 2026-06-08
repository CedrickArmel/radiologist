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

import os
from typing import Optional, Sequence

import rich
import rich.syntax
import rich.tree
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt

from radiologist.utils.ml.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print a Hydra DictConfig as a rich tree.

    Args:
        cfg: The Hydra config to print.
        print_order: Fields to show first; remaining fields appended in natural order.
        resolve: If True, resolve OmegaConf interpolations before printing.
        save_to_file: If True, also write to cfg.paths.output_dir/config_tree.log.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    ordered_keys = list(print_order) + [k for k in cfg if k not in print_order]

    for field in ordered_keys:
        if field not in cfg:
            logger.debug(
                f"Field '{field}' not found in config. Skipping '{field}' config printing..."
            )
            continue
        branch = tree.add(str(field), style=style, guide_style=style)
        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_to_file:
        output_dir: Optional[str] = OmegaConf.select(cfg, "paths.output_dir")
        if output_dir is not None:
            with open(os.path.join(output_dir, "config_tree.log"), "w") as f:
                rich.print(tree, file=f)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompt for tags when empty; raise during multirun if tags are missing.

    Args:
        cfg: The Hydra config.
        save_to_file: If True, write tags to cfg.paths.output_dir/config_tree.log.

    Raises:
        ValueError: During a multirun when cfg.tags is empty or None.
    """
    tags: Optional[Sequence[str]] = OmegaConf.select(cfg, "tags")
    has_tags = bool(tags)

    choices: Optional[DictConfig] = OmegaConf.select(cfg, "hydra.runtime.choices")
    is_multirun = choices is not None and "multirun" in list(choices.values())

    if is_multirun and not has_tags:
        raise ValueError(
            "Tags must be provided for multirun experiments. "
            "Set `tags` in your config to identify this sweep."
        )

    if not has_tags:
        logger.warning("No tags provided in config. Prompting user to input tags...")
        raw = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in raw.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

    if save_to_file:
        output_dir: Optional[str] = OmegaConf.select(cfg, "paths.output_dir")
        if output_dir is not None:
            with open(os.path.join(output_dir, "config_tree.log"), "a") as f:
                f.write(f"tags: {list(tags or [])}\n")
