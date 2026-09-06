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

"""Behavioral tests for issue #231's nightly/pre-release minimum-version job.

Every intra-workspace and third-party requirement in this repo declares a
lower bound (``>=0.1.0``, ``>=2.2.0``, …) that CI never actually exercises:
`ci.yml`/`publish.yml`'s `test` job always resolves to the newest compatible
version via `uv.lock`. This issue adds a new, standalone reusable workflow,
`.github/workflows/min-versions.yml`, that resolves the workspace DOWN to
each direct dependency's declared floor (`uv sync --resolution
lowest-direct`) and runs the full suite against it -- on a nightly schedule
and, via `publish.yml`'s new `min-versions` job, immediately before a
publish. It must never run on an ordinary pull request.

These tests parse the workflow YAML as text (no ``pyyaml`` dependency),
mirroring `scripts_tests/test_ci_workflow_toolchain.py` and
`scripts_tests/test_publish_testpypi_dry_run.py` (test files in this repo
don't import each other -- the `_job_lines` helper is copied, not shared).
"""

import re
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"
_MIN_VERSIONS_WORKFLOW = _WORKFLOWS_DIR / "min-versions.yml"
_PUBLISH_WORKFLOW = _WORKFLOWS_DIR / "publish.yml"

_JOB_HEADER_RE = re.compile(r"^  (\S+):\s*$")


def _min_versions_text() -> str:
    return _MIN_VERSIONS_WORKFLOW.read_text()


def _publish_text() -> str:
    return _PUBLISH_WORKFLOW.read_text()


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


def test_min_versions_workflow_file_exists() -> None:
    assert _MIN_VERSIONS_WORKFLOW.is_file()


def test_min_versions_runs_on_a_nightly_schedule() -> None:
    text = _min_versions_text()
    assert re.search(r"^\s*schedule:\s*$", text, flags=re.MULTILINE)
    assert re.search(r"cron:\s*[\"'][^\"']+[\"']", text)


def test_min_versions_can_be_triggered_manually() -> None:
    assert re.search(
        r"^\s*workflow_dispatch:\s*$", _min_versions_text(), flags=re.MULTILINE
    )


def test_min_versions_is_callable_as_a_reusable_workflow() -> None:
    text = _min_versions_text()
    assert re.search(r"^\s*workflow_call:\s*$", text, flags=re.MULTILINE)
    assert "ref:" in text


def test_min_versions_never_triggers_on_pull_request() -> None:
    assert not re.search(r"^\s*pull_request:", _min_versions_text(), flags=re.MULTILINE)
    assert not re.search(r"^\s*push:", _min_versions_text(), flags=re.MULTILINE)


def test_min_versions_job_checks_out_the_requested_ref() -> None:
    lines = _job_lines(_min_versions_text(), "lowest-direct")
    block = "\n".join(lines)
    assert "uses: actions/checkout" in block
    assert "${{ inputs.ref }}" in block


def test_min_versions_job_uses_local_setup_uv_with_locked_false() -> None:
    """The whole point of this job is to resolve *away* from uv.lock."""
    lines = _job_lines(_min_versions_text(), "lowest-direct")
    block = "\n".join(lines)
    assert "uses: ./.github/actions/setup-uv" in block
    assert 'locked: "false"' in block


def test_min_versions_job_checks_out_before_the_local_composite_action() -> None:
    lines = _job_lines(_min_versions_text(), "lowest-direct")
    composite_index = next(
        index
        for index, line in enumerate(lines)
        if "uses: ./.github/actions/setup-uv" in line
    )
    preceding = "\n".join(lines[:composite_index])
    assert "uses: actions/checkout" in preceding


def test_min_versions_job_installs_the_lowest_direct_resolution() -> None:
    lines = _job_lines(_min_versions_text(), "lowest-direct")
    start = next(
        index
        for index, line in enumerate(lines)
        if "uv sync" in line and "--active" in line
    )
    end = start
    while lines[end].rstrip().endswith("\\"):
        end += 1
    install_command = " ".join(line.strip() for line in lines[start : end + 1])
    assert "--resolution lowest-direct" in install_command
    assert (
        "--no-extra ray" in install_command
    ), "the deferred Ray backend must stay excluded"


def test_min_versions_job_runs_make_test_without_resyncing() -> None:
    """`uv run` re-syncs before running -- that would silently pull the
    environment back up to `uv.lock`'s newest versions, defeating the whole
    job. `UV_NO_SYNC` must gate the `make test` step specifically."""
    lines = _job_lines(_min_versions_text(), "lowest-direct")
    block = "\n".join(lines)
    test_step_index = next(
        index
        for index, line in enumerate(lines)
        if re.search(r"run:\s*make test", line)
    )
    # The UV_NO_SYNC env must be associated with this step, i.e. appear
    # between this step's dash and the next step's dash (or end of job).
    following_dash_indices = [
        index
        for index, line in enumerate(lines)
        if index > test_step_index and re.match(r"^\s*-\s", line)
    ]
    end = following_dash_indices[0] if following_dash_indices else len(lines)
    step_block = "\n".join(lines[max(0, test_step_index - 3) : end])
    assert "make test" in block
    assert 'UV_NO_SYNC: "true"' in step_block


def test_min_versions_job_never_uploads_commits_or_pushes_anything() -> None:
    lines = _job_lines(_min_versions_text(), "lowest-direct")
    block = "\n".join(lines)
    assert "upload-artifact" not in block
    assert "git push" not in block
    assert "git commit" not in block
    assert "actions/checkout" in block  # sanity: the job block was found


def test_min_versions_workflow_declares_read_only_permissions() -> None:
    text = _min_versions_text()
    assert re.search(r"^permissions:\s*$", text, flags=re.MULTILINE)
    assert re.search(r"^\s*contents:\s*read\s*$", text, flags=re.MULTILINE)


def test_publish_gains_a_min_versions_job_gating_the_release() -> None:
    lines = _job_lines(_publish_text(), "min-versions")
    block = "\n".join(lines)
    assert "needs: resolve" in block or re.search(
        r"needs:\s*\[[^\]]*resolve[^\]]*\]", block
    )
    assert "uses: ./.github/workflows/min-versions.yml" in block
    assert "${{ needs.resolve.outputs.sha }}" in block


def test_publish_build_job_now_also_needs_min_versions() -> None:
    block = "\n".join(_job_lines(_publish_text(), "build"))
    needs_line = next(line for line in block.splitlines() if "needs:" in line)
    assert "resolve" in needs_line
    assert "test" in needs_line
    assert "min-versions" in needs_line
