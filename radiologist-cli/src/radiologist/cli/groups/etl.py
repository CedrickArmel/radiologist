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

"""``radiologist etl`` command group — Hydra-composed ETL pipeline entry point."""

from typing import List

import hydra
from omegaconf import DictConfig

__all__ = ["etl_main", "run"]


@hydra.main(
    config_path="pkg://radiologist.etl.conf",
    config_name="etl",
    version_base=None,
)
def etl_main(cfg: DictConfig) -> None:
    """Hydra-composed entry point for the ``radiologist etl`` command group.

    Args:
        cfg: fully composed Hydra ``DictConfig``, injected by the
            ``@hydra.main`` decorator — never passed explicitly by callers.
    """
    raise NotImplementedError


def run(argv: List[str]) -> int:
    """Run the ``etl`` group with ``argv`` as the effective ``sys.argv[1:]``.

    Args:
        argv: Arguments forwarded from the dispatcher, after the ``etl``
            group token has been stripped.

    Returns:
        The process exit code.
    """
    raise NotImplementedError
