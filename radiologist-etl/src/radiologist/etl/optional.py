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

"""Optional third-party import guard shared across the etl package.

Wraps the ``prefect`` import in ``try/except ImportError`` so the package
imports cleanly without the extra installed, mirroring the ``_wandb``
sentinel pattern in ``radiologist.registry.optional`` /
``radiologist.inference.optional``. Unlike those, prefect's absence is not a
hard error here: ``prefect_pipelines.py`` degrades to a warning-only,
unrecorded flow rather than raising, so this module exposes no-op stand-ins
for the Prefect API surface it uses, not just an availability flag.
"""

try:
    from prefect import flow, task
    from prefect.artifacts import (
        create_link_artifact,
        create_markdown_artifact,
        create_table_artifact,
    )
    from prefect.cache_policies import INPUTS
    from prefect.task_runners import ProcessPoolTaskRunner, TaskRunner, unmapped

    _PREFECT_AVAILABLE = True
    _PREFECT_IMPORT_ERROR = ""
except ImportError as ex:  # pragma: no cover

    def flow(fn=None, **_):  # type: ignore[misc, no-redef]
        """No-op stand-in for ``prefect.flow`` when prefect is not installed."""
        return fn if fn is not None else (lambda f: f)

    def task(fn=None, **_):  # type: ignore[misc, no-redef]
        """No-op stand-in for ``prefect.task`` when prefect is not installed."""
        return fn if fn is not None else (lambda f: f)

    def create_link_artifact(**_):  # type: ignore[misc, no-redef]
        """No-op stand-in for ``prefect.artifacts.create_link_artifact``."""

    def create_markdown_artifact(**_):
        """No-op stand-in for ``prefect.artifacts.create_markdown_artifact``."""

    def create_table_artifact(**_):  # type: ignore[misc, no-redef]
        """No-op stand-in for ``prefect.artifacts.create_table_artifact``."""

    def unmapped(value):  # type: ignore[misc, no-redef]
        """No-op stand-in for ``prefect.tasks.unmapped`` when prefect is not installed."""
        return value

    ProcessPoolTaskRunner = None  # type: ignore[assignment, misc]
    TaskRunner = None  # type: ignore[assignment, misc]

    _PREFECT_AVAILABLE = False
    _PREFECT_IMPORT_ERROR = str(ex)
    INPUTS = None  # type: ignore[assignment]

_PREFECT_MISSING_MSG = (
    "prefect is required to record etl runs. "
    "Install with: pip install 'radiologist-etl[prefect]'"
)

try:
    import prefect_dask  # type: ignore[import-untyped]  # noqa: F401

    _PREFECT_DASK_AVAILABLE = True
except ImportError:
    _PREFECT_DASK_AVAILABLE = False

try:
    import prefect_ray  # type: ignore[import-untyped]  # noqa: F401

    _PREFECT_RAY_AVAILABLE = True
except ImportError:
    _PREFECT_RAY_AVAILABLE = False

try:
    import apache_beam  # type: ignore[import-untyped]  # noqa: F401

    _BEAM_AVAILABLE = True
except ImportError:
    _BEAM_AVAILABLE = False
