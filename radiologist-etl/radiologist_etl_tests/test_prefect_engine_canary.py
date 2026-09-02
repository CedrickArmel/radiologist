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

"""Regression canary for a real (non-``.fn``) Prefect flow run.

Every other Prefect-touching test in this suite deliberately bypasses
Prefect's orchestration layer via the documented ``.fn`` escape hatch (see
``test_prefect_pipelines.py``'s module docstring) because this environment's
embedded local/ephemeral Prefect API server is incompatible with newer
FastAPI internals (``AttributeError: 'PrefectRouter' object has no
attribute 'routes'``, see GitHub issue #192). That bypass is correct for
testing ETL business logic, but it structurally cannot catch a regression
in the Prefect/FastAPI/Starlette dependency triangle itself, since it never
exercises the real routing code path.

This test intentionally does the opposite: it runs a trivial ``@flow`` for
real, forcing the local/ephemeral server path, so any future dependency
drift on this axis fails here — one clear, obviously-diagnosable test —
instead of manifesting as confusing, unrelated-looking failures scattered
across the suite.
"""

from __future__ import annotations

import pytest

from radiologist.etl.optional import _PREFECT_AVAILABLE

pytestmark = pytest.mark.skipif(
    not _PREFECT_AVAILABLE, reason="prefect extra not installed"
)


def test_real_flow_run_completes_against_local_ephemeral_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real (non-``.fn``) flow run must complete and return its result.

    Forces Prefect's local/ephemeral API server path by clearing
    ``PREFECT_API_URL``/``PREFECT_API_KEY`` (rather than pointing at a real
    backend), then invokes a trivial ``@flow``-decorated function as a
    caller would — through Prefect's orchestration layer, not through
    ``.fn`` — so this test genuinely exercises the routing code path that
    broke in issue #192.
    """
    monkeypatch.setenv("PREFECT_API_URL", "")
    monkeypatch.setenv("PREFECT_API_KEY", "")

    from prefect import flow

    @flow
    def _trivial_flow() -> int:
        return 1

    result = _trivial_flow()

    assert result == 1
