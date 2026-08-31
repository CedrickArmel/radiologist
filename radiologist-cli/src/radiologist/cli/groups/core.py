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

"""``radiologist core`` command group — Hydra-composed training entry point."""

from typing import List, Optional

import hydra
from omegaconf import DictConfig

__all__ = ["train_main", "run"]


@hydra.main(
    config_path="pkg://radiologist.core.configs",
    config_name="train",
    version_base="1.3",
)
def train_main(cfg: DictConfig) -> Optional[float]:
    """Hydra-composed entry point for the ``radiologist core`` command group.

    Args:
        cfg: fully composed Hydra ``DictConfig``, injected by the
            ``@hydra.main`` decorator — never passed explicitly by callers.

    Returns:
        The scalar value of ``cfg.optimized_metric``, or ``None`` when unset.
    """
    raise NotImplementedError


def run(argv: List[str]) -> int:
    """Run the ``core`` group with ``argv`` as the effective ``sys.argv[1:]``.

    Args:
        argv: Arguments forwarded from the dispatcher, after the ``core``
            group token has been stripped.

    Returns:
        The process exit code.
    """
    raise NotImplementedError
