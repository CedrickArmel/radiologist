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

"""Behavioral tests for issue #226's `setup-uv` composite-action adoption.

`.github/actions/setup-uv/action.yml` (issue #225's skeleton) is a Python+uv
toolchain installer that exports `UV_LOCKED`, deliberately narrower than
`.github/actions/setup-and-test` (which owns checkout + sync + `make test`
for the two `test` jobs). Before this issue, every *other* job across
`ci.yml`, `publish.yml`, `release.yml` and `docs.yml` hand-rolled the same
`actions/setup-python` + `astral-sh/setup-uv` pair with a floating major-
version tag -- a supply-chain risk given `publish.yml`'s `publish` job holds
`id-token: write`.

This issue is a pure substitution: same toolchain, same Python/uv versions,
same commands run afterwards. It does not change `UV_LOCKED` enforcement
(every call site here passes `locked: "false"`; issue #228 flips that) and
does not fold `setup-and-test` onto `setup-uv` (a documented, deliberate,
temporary two-line duplication).

These tests parse the workflow/action YAML as text (no `pyyaml` dependency),
mirroring `scripts_tests/test_ci_workflows_exclude_ray.py`.
"""

import re
from pathlib import Path
from typing import List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"
_SETUP_UV_ACTION = _REPO_ROOT / ".github" / "actions" / "setup-uv" / "action.yml"

_JOB_HEADER_RE = re.compile(r"^  (\S+):\s*$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Every non-test job that hand-rolled the toolchain pair before this issue.
# `publish.yml`'s `resolve` job is included: it also called
# `astral-sh/setup-uv@v3` directly and the acceptance criteria forbid any
# remaining direct reference, not just the ones in the issue's Before/After
# illustrations.
_TARGET_JOBS: List[Tuple[str, str]] = [
    ("ci.yml", "build"),
    ("publish.yml", "resolve"),
    ("publish.yml", "build"),
    ("publish.yml", "publish"),
    ("publish.yml", "tag"),
    ("release.yml", "bump"),
    ("docs.yml", "build"),
]


def _read_workflow(name: str) -> str:
    return (_WORKFLOWS_DIR / name).read_text()


def _workflow_texts() -> List[str]:
    return [path.read_text() for path in _WORKFLOWS_DIR.glob("*.yml")]


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


@pytest.mark.parametrize("workflow, job", _TARGET_JOBS)
def test_target_job_adopts_the_local_setup_uv_action(workflow: str, job: str) -> None:
    block = "\n".join(_job_lines(_read_workflow(workflow), job))
    assert "uses: ./.github/actions/setup-uv" in block
    assert "actions/setup-python@" not in block
    assert "astral-sh/setup-uv@" not in block


@pytest.mark.parametrize("workflow, job", _TARGET_JOBS)
def test_target_job_checks_out_before_the_local_composite_action(
    workflow: str, job: str
) -> None:
    """A local `uses: ./...` action can't be resolved before a checkout."""
    lines = _job_lines(_read_workflow(workflow), job)
    composite_index = next(
        index
        for index, line in enumerate(lines)
        if "uses: ./.github/actions/setup-uv" in line
    )
    preceding = "\n".join(lines[:composite_index])
    assert "uses: actions/checkout" in preceding, (
        f"{workflow}:{job} must check out the repo before invoking the "
        "local setup-uv composite action"
    )


@pytest.mark.parametrize("workflow, job", _TARGET_JOBS)
def test_target_job_keeps_this_slice_behaviourless(workflow: str, job: str) -> None:
    """`UV_LOCKED` enforcement is issue #228's job, not this one's."""
    block = "\n".join(_job_lines(_read_workflow(workflow), job))
    assert 'locked: "false"' in block


def test_no_workflow_references_the_third_party_actions_directly() -> None:
    for text in _workflow_texts():
        assert "actions/setup-python@" not in text
        assert "astral-sh/setup-uv@" not in text


@pytest.mark.parametrize("workflow", ["ci.yml", "publish.yml"])
def test_test_job_remains_untouched(workflow: str) -> None:
    block = "\n".join(_job_lines(_read_workflow(workflow), "test"))
    assert "uses: ./.github/actions/setup-and-test" in block
    assert "uses: ./.github/actions/setup-uv" not in block


def test_publish_job_gains_a_checkout_scoped_to_the_github_directory() -> None:
    """`publish.yml`'s `publish` job previously had no checkout at all."""
    block = "\n".join(_job_lines(_read_workflow("publish.yml"), "publish"))
    assert "uses: actions/checkout" in block
    assert "sparse-checkout: .github" in block


def test_publish_job_installs_no_third_party_action_beyond_uv_and_transfer() -> None:
    """The `id-token: write` job must not widen its third-party footprint."""
    lines = _job_lines(_read_workflow("publish.yml"), "publish")
    uses_lines = [line.strip() for line in lines if line.strip().startswith("- uses:")]
    assert uses_lines, "expected at least one 'uses:' step"
    for line in uses_lines:
        assert (
            "./.github/actions/setup-uv" in line
            or "actions/checkout" in line
            or "actions/download-artifact" in line
        ), line


def test_publish_job_does_not_run_setup_python() -> None:
    block = "\n".join(_job_lines(_read_workflow("publish.yml"), "publish"))
    assert 'setup-python: "false"' in block


@pytest.mark.parametrize(
    "workflow, job",
    [("publish.yml", "resolve"), ("publish.yml", "tag"), ("release.yml", "bump")],
)
def test_uv_only_jobs_skip_setup_python(workflow: str, job: str) -> None:
    """These jobs never ran `actions/setup-python` before -- keep it that way."""
    block = "\n".join(_job_lines(_read_workflow(workflow), job))
    assert 'setup-python: "false"' in block


def test_docs_workflow_install_step_remains_byte_for_byte_unchanged() -> None:
    docs_text = _read_workflow("docs.yml")
    assert "- run: uv sync --group docs --all-packages --all-extras" in docs_text


def test_setup_uv_action_pins_each_wrapped_action_to_one_commit_sha() -> None:
    lines = _SETUP_UV_ACTION.read_text().splitlines()
    setup_python_lines = [
        line for line in lines if "uses: actions/setup-python@" in line
    ]
    setup_uv_lines = [line for line in lines if "uses: astral-sh/setup-uv@" in line]
    assert len(setup_python_lines) == 1
    assert len(setup_uv_lines) == 1
    for line in (setup_python_lines[0], setup_uv_lines[0]):
        sha = line.split("@", 1)[1].split()[0]
        assert _SHA_RE.match(sha), f"expected a 40-char commit sha, got {sha!r}"
        assert "#" in line, f"{line!r} is missing a trailing release-tag comment"


def test_setup_and_test_action_is_left_untouched() -> None:
    """This issue must not fold `setup-and-test`'s responsibilities away."""
    setup_and_test = (
        _REPO_ROOT / ".github" / "actions" / "setup-and-test" / "action.yml"
    )
    text = setup_and_test.read_text()
    assert "actions/checkout" in text
    assert "uv sync" in text
    assert "make test" in text
