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

import logging
from typing import Any, Mapping, MutableMapping, Optional, Tuple

from lightning_utilities.core.rank_zero import rank_zero_only as _rank_zero_only  # type: ignore[import-untyped]


class RankedLogger(logging.LoggerAdapter):
    """Logger adapter that prefixes messages with the distributed process rank.

    Args:
        name: Logger name.
        rank_zero_only: When True, only rank-0 messages are emitted.
        extra: Additional context merged into every log record.
    """

    def __init__(
        self,
        name: str = __name__,
        rank_zero_only: bool = False,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger, extra or {})
        self._rank_zero_only = rank_zero_only

    def log(  # type: ignore[override]
        self,
        level: int,
        msg: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        rank: int = getattr(_rank_zero_only, "rank", 0)

        if self._rank_zero_only and rank != 0:
            return

        if self.isEnabledFor(level):
            prefix = f"[rank {rank}] "
            full_msg, kwargs = self.process(f"{prefix}{msg}", kwargs)  # type: ignore[arg-type,assignment]
            self.logger.log(level, full_msg, *args, **kwargs)

    def process(
        self, msg: str, kwargs: MutableMapping[str, Any]
    ) -> Tuple[str, MutableMapping[str, Any]]:
        return super().process(msg, kwargs)
