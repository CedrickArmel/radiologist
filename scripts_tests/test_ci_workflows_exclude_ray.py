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

These tests parse the workflow YAML as text (no ``pyyaml`` dependency) so
they stay green on a checkout where no extra at all is installed.
"""

import re
from pathlib import Path
from typing import List

_WORKFLOWS_DIR = Path(__file__).resolve().parents[1] / ".github" / "workflows"

_JOB_HEADER_RE = re.compile(r"^  (\S+):\s*$")


def _read_workflow(name: str) -> str:
    return (_WORKFLOWS_DIR / name).read_text()


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


def test_ci_test_job_install_step_excludes_ray() -> None:
    lines = _job_lines(_read_workflow("ci.yml"), "test")
    install_line = _install_step_line(lines)
    assert "--no-extra ray" in install_line


def test_publish_test_job_install_step_excludes_ray() -> None:
    lines = _job_lines(_read_workflow("publish.yml"), "test")
    install_line = _install_step_line(lines)
    assert "--no-extra ray" in install_line


def test_ci_and_publish_test_jobs_install_the_same_dependency_set() -> None:
    ci_install = _install_step_line(_job_lines(_read_workflow("ci.yml"), "test"))
    publish_install = _install_step_line(
        _job_lines(_read_workflow("publish.yml"), "test")
    )
    assert ci_install == publish_install


def test_ray_exclusion_reason_is_documented_next_to_the_install_step() -> None:
    for workflow in ("ci.yml", "publish.yml"):
        lines = _job_lines(_read_workflow(workflow), "test")
        install_index = _install_step_index(lines)
        preceding = "\n".join(lines[max(0, install_index - 3) : install_index]).lower()
        assert (
            "ray" in preceding
        ), f"{workflow}: no comment mentions ray above the install step"
        assert (
            "188" in preceding
        ), f"{workflow}: no comment references issue #188 above the install step"


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
