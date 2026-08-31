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

"""Sentinel handles for the optional business-package extras.

Each name is set to the imported module when its ``radiologist-cli`` extra is
installed, or ``None`` otherwise. ``require()`` turns an absent or
feature-incomplete module into a clear ``RuntimeError`` naming the missing
``radiologist-cli`` extra.
"""

from types import ModuleType
from typing import Optional

_etl: Optional[ModuleType]
_registry: Optional[ModuleType]
_inference: Optional[ModuleType]

try:
    import radiologist.etl as _etl  # type: ignore[import-untyped,no-redef]
except ImportError:
    _etl = None

try:
    import radiologist.registry as _registry  # type: ignore[import-untyped,no-redef]
except ImportError:
    _registry = None

try:
    import radiologist.inference as _inference  # type: ignore[import-untyped,no-redef]
except ImportError:
    _inference = None

__all__ = ["require"]


def require(extra: str) -> ModuleType:
    """Return the business-package module backing ``extra``, or raise.

    Args:
        extra: One of ``"etl"``, ``"registry"``, ``"inference"``.

    Returns:
        The imported module.

    Raises:
        RuntimeError: When the module is absent, or when its feature-level
            sentinel is unavailable (``registry`` ->
            ``radiologist.registry.optional._wandb``, ``etl`` ->
            ``radiologist.etl.optional._PREFECT_AVAILABLE``,
            ``inference`` -> module importability only), naming
            ``pip install 'radiologist-cli[<extra>]'``.
    """
    hint = f"pip install 'radiologist-cli[{extra}]'"
    module = {"etl": _etl, "registry": _registry, "inference": _inference}.get(extra)
    if module is None:
        raise RuntimeError(hint)

    if extra == "etl":
        from radiologist.etl import optional as etl_optional

        if not etl_optional._PREFECT_AVAILABLE:
            raise RuntimeError(hint)
    elif extra == "registry":
        from radiologist.registry import optional as registry_optional

        if registry_optional._wandb is None:
            raise RuntimeError(hint)

    return module
