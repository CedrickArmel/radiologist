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

"""Behavioral tests for the CI/publish workflows' Ray-exclusion contract.

See issue #216: `.github/workflows/ci.yml`'s and `.github/workflows/publish.yml`'s
``test`` job both install the workspace with ``uv sync --all-extras`` before
running the suite. ``--all-extras`` enumerates every named extra of every
workspace package and ignores extra composition, so both jobs force-install
the still-under-development Ray execution backend (issue #188). Both jobs
must exclude it identically, while ``docs.yml`` -- which needs every module
importable, including Ray-referencing Hydra config targets -- must keep
installing it, untouched.

Issue #167 later extracted the shared install/test steps out of each
workflow's ``test`` job into the composite action
``.github/actions/setup-and-test/action.yml``, so the ray-exclusion install
line now lives in that single shared file instead of being duplicated
inline in each workflow. This is a deliberate, explicit deviation from
issue #167's own "no test may be changed" acceptance criterion -- the user
decided, after being shown that the extraction could not otherwise leave
these tests passing unmodified, to relocate what these tests check rather
than block the refactor or leave the install command duplicated. The
assertions below preserve the same behavioral coverage as before, just
pointed at the new location.

These tests parse the workflow/action YAML as text (no ``pyyaml``
dependency) so they stay green on a checkout where no extra at all is
installed.
"""

import re
from pathlib import Path
from typing import List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"
_COMPOSITE_ACTION = _REPO_ROOT / ".github" / "actions" / "setup-and-test" / "action.yml"

_JOB_HEADER_RE = re.compile(r"^  (\S+):\s*$")


def _read_workflow(name: str) -> str:
    return (_WORKFLOWS_DIR / name).read_text()


def _read_composite_action() -> str:
    return _COMPOSITE_ACTION.read_text()


def _composite_action_lines() -> List[str]:
    return _read_composite_action().splitlines()


def _job_lines(workflow_text: str, job_name: str) -> List[str]:
    """Return the raw lines belonging to a single top-level job block."""
    lines = workflow_text.splitlines()
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


def _install_step_index(job_lines: List[str]) -> int:
    """Return the index, within ``job_lines``, of the ``uv sync`` install step."""
    candidates = [
        index
        for index, line in enumerate(job_lines)
        if "uv sync" in line and "--all-extras" in line
    ]
    assert len(candidates) == 1, (
        "expected exactly one 'uv sync ... --all-extras' install step, got "
        f"{[job_lines[i] for i in candidates]!r}"
    )
    return candidates[0]


def _install_step_line(job_lines: List[str]) -> str:
    return job_lines[_install_step_index(job_lines)].strip()


@pytest.mark.parametrize("workflow", ["ci.yml", "publish.yml"])
def test_test_job_delegates_to_the_shared_setup_and_test_action(
    workflow: str,
) -> None:
    """Each workflow's `test` job installs/tests via the composite action.

    The install command itself no longer lives inline in the job body (see
    module docstring) -- what each job body must still show is that it calls
    the one shared action rather than re-declaring its own steps.
    """
    lines = _job_lines(_read_workflow(workflow), "test")
    block = "\n".join(lines)
    assert "uses: ./.github/actions/setup-and-test" in block
    assert "uv sync" not in block, (
        f"{workflow}: the install command should live only in the composite "
        "action, not duplicated inline in the job body"
    )


def test_ci_and_publish_test_jobs_call_the_identical_composite_action() -> None:
    """The two `test` jobs reference the exact same action, byte-for-byte.

    Replaces the old "install the same dependency set" check: since the
    install command now lives in a single shared file, the dependency set is
    identical by construction. What can still drift is *which* action --
    or which pinned ref of it -- each workflow calls.
    """

    def _uses_line(workflow: str) -> str:
        lines = _job_lines(_read_workflow(workflow), "test")
        candidates = [
            line.strip() for line in lines if "uses: ./.github/actions" in line
        ]
        assert len(candidates) == 1
        return candidates[0]

    assert _uses_line("ci.yml") == _uses_line("publish.yml")


def test_composite_action_install_step_excludes_ray() -> None:
    lines = _composite_action_lines()
    install_line = _install_step_line(lines)
    assert "--no-extra ray" in install_line


def _preceding_comment_block(job_lines: List[str], step_index: int) -> str:
    """Return the contiguous run of ``#``-comment lines directly above a step."""
    start = step_index
    while start > 0 and job_lines[start - 1].strip().startswith("#"):
        start -= 1
    return "\n".join(job_lines[start:step_index]).lower()


def test_ray_exclusion_reason_is_documented_next_to_the_install_step() -> None:
    lines = _composite_action_lines()
    preceding = _preceding_comment_block(lines, _install_step_index(lines))
    assert "ray" in preceding, "no comment mentions ray above the install step"
    assert "188" in preceding, "no comment references issue #188 above the install step"


def test_publish_build_job_still_gates_on_the_test_job() -> None:
    lines = _job_lines(_read_workflow("publish.yml"), "build")
    block = "\n".join(lines)
    assert re.search(
        r"needs:\s*\[[^\]]*\btest\b[^\]]*\]", block
    ), "publish.yml's build job must still depend on the test job"


def test_publish_test_job_is_not_weakened() -> None:
    lines = _job_lines(_read_workflow("publish.yml"), "test")
    block = "\n".join(lines)
    assert "continue-on-error" not in block
    assert not re.search(r"^\s*if:", block, flags=re.MULTILINE)


def test_docs_workflow_install_step_is_byte_for_byte_unchanged() -> None:
    docs_text = _read_workflow("docs.yml")
    assert "- run: uv sync --group docs --all-packages --all-extras" in docs_text
    assert "--no-extra ray" not in docs_text
