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

"""Optional third-party import guards shared across the registry package.

Wraps the `wandb` and `typer` imports in `try/except ImportError` so the
package imports cleanly without either extra installed; callers that need
the real SDK invoke `_guard_wandb()` first to fail with a clear message.
"""

try:
    import wandb as _wandb  # type: ignore[import-untyped]
except ImportError:
    _wandb = None  # type: ignore[assignment]

_WANDB_MISSING_MSG = (
    "wandb is required for registry operations. "
    "Install with: pip install 'radiologist-registry[wandb]'"
)

_MODEL_ARTIFACT_TYPE = "model"


def _guard_wandb() -> None:
    """Raise a clear error if the `wandb` extra is not installed.

    Raises:
        RuntimeError: If `wandb` could not be imported.
    """
    if _wandb is None:
        raise RuntimeError(_WANDB_MISSING_MSG)


try:
    import typer as _typer  # type: ignore[import-untyped]
except ImportError:
    _typer = None  # type: ignore[assignment]

_TYPER_MISSING_MSG = (
    "typer is required for the radiologist-registry CLI. "
    "Install with: pip install 'radiologist-registry[cli]'"
)
