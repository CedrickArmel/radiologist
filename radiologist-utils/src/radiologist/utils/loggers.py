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