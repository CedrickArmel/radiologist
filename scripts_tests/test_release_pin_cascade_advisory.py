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

"""Behavioral tests for issue #233's advisory pin-cascade report step.

``release.yml``'s ``bump`` job now posts a non-blocking comment on the
release pull request it just opened, listing dependents whose pin floor is
now stale relative to the distribution being bumped. This must never fail
the release: the step is tolerant of its own failure.

These tests parse the workflow YAML as text (no ``pyyaml`` dependency),
mirroring ``scripts_tests/test_publish_testpypi_dry_run.py``.
"""

import re
from pathlib import Path

from _workflow_test_helpers import _job_lines

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"


def _release_text() -> str:
    return _RELEASE_WORKFLOW.read_text()


def _bump_block() -> str:
    return "\n".join(_job_lines(_release_text(), "bump"))


def test_bump_job_posts_a_pin_cascade_advisory_step() -> None:
    block = _bump_block()
    assert "Post the advisory pin-cascade report" in block


def test_pin_cascade_step_is_placed_after_the_pull_request_is_opened() -> None:
    lines = _job_lines(_release_text(), "bump")
    pr_index = next(
        index
        for index, line in enumerate(lines)
        if "Open the release pull request" in line
    )
    advisory_index = next(
        index
        for index, line in enumerate(lines)
        if "Post the advisory pin-cascade report" in line
    )
    assert advisory_index > pr_index


def test_pin_cascade_step_never_fails_the_job() -> None:
    lines = _job_lines(_release_text(), "bump")
    advisory_index = next(
        index
        for index, line in enumerate(lines)
        if "Post the advisory pin-cascade report" in line
    )
    end = len(lines)
    for index in range(advisory_index + 1, len(lines)):
        if re.match(r"^      - name:", lines[index]):
            end = index
            break
    block = "\n".join(lines[advisory_index:end])
    assert "continue-on-error: true" in block


def test_pin_cascade_step_invokes_the_stale_pins_markdown_cli() -> None:
    block = _bump_block()
    assert "workspace_manifests.py stale-pins" in block
    assert "--format markdown" in block


def test_pin_cascade_step_posts_an_idempotent_pull_request_comment() -> None:
    block = _bump_block()
    assert "gh pr comment" in block
    assert "--edit-last" in block
    assert "--create-if-none" in block


def test_pin_cascade_step_targets_the_release_branch_just_created() -> None:
    block = _bump_block()
    advisory_start = block.index("Post the advisory pin-cascade report")
    advisory_block = block[advisory_start:]
    assert "steps.branch.outputs.name" in advisory_block
