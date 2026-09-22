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

"""Shared plain-text YAML parsing helpers for ``scripts_tests/`` workflow tests.

Several sibling test files each independently parse ``.github/workflows/*.yml``
and ``.github/actions/*/action.yml`` as plain text (no ``pyyaml`` dependency,
so these tests stay green on a checkout where no extra at all is installed).
``_job_lines``/``_JOB_HEADER_RE`` had accreted six byte-identical (or
near-identical) copies across those files. This module is the one shared
home for them.

Not named ``conftest.py``: this flat (non-package) test directory has no
``__init__.py``, so a bare ``import conftest`` here would collide with the
repository-root ``conftest.py`` under Python's own module-name resolution
(pytest itself loads each ``conftest.py`` by file path, sidestepping this,
but a plain importable module cannot). This file is not named ``test_*.py``
either, so pytest does not collect it as a test module -- importing from it
is not the "test files import each other" pattern this repo otherwise
avoids.
"""

import re
from typing import List

_JOB_HEADER_RE = re.compile(r"^  (\S+):\s*$")


def _job_lines(workflow_text: str, job_name: str) -> List[str]:
    """Return the raw lines belonging to a single top-level job block.

    Slices from the ``jobs:`` line onward first, so a workflow's ``on:``
    block keys (e.g. ``pull_request:``, ``workflow_dispatch:``), which share
    the same 2-space indent as a real job name, never get misidentified as
    one (see issue #234).
    """
    lines = workflow_text.splitlines()
    jobs_start = next(
        (index for index, line in enumerate(lines) if line.strip() == "jobs:"),
        0,
    )
    lines = lines[jobs_start:]
    start = None
    for index, line in enumerate(lines):
        match = _JOB_HEADER_RE.match(line)
        if match and match.group(1) == job_name:
            start = index + 1
            break
    if start is None:
        raise AssertionError(f"job {job_name!r} not found")
    end = len(lines)
    for index in range(start, len(lines)):
        if _JOB_HEADER_RE.match(lines[index]):
            end = index
            break
    return lines[start:end]
