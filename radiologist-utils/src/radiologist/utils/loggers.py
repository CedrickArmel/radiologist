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

import logging
from typing import Mapping, Optional


class Logger(logging.LoggerAdapter):
    """A plain python command line logger."""

    def __init__(
        self,
        name: str = __name__,
        extra: Optional[Mapping[str, object]] = None,
    ) -> None:
        logger = logging.getLogger(name)
        super().__init__(logger=logger, extra=extra)

    def log(  # type: ignore[override]
        self, level: int, msg: str, *args, **kwargs
    ) -> None:
        if self.isEnabledFor(level):
            msg, kwargs = self.process(msg, kwargs)  # type: ignore[assignment]
            self.logger.log(level, msg, *args, **kwargs)
