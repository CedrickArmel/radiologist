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

"""Behavioral tests for issue #232's real-PyPI resolution guard.

`.github/workflows/publish.yml` never checks whether the distribution being
published can actually be installed from the public index -- inside this
workspace `[tool.uv.sources] { workspace = true }` silently satisfies an
intra-workspace requirement that PyPI itself cannot. This adds a
`resolution-guard` job, inserted between `build` and `publish-testpypi`,
that resolves `workspace_manifests.publishable_requirement_lines(...)` with
`uv pip compile --no-sources` against the real index.

These tests parse the workflow YAML as text (no `pyyaml` dependency),
mirroring `scripts_tests/test_publish_testpypi_dry_run.py` and
`scripts_tests/test_ci_workflows_exclude_ray.py`.
"""

from pathlib import Path

from _workflow_test_helpers import _job_lines

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PUBLISH_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "publish.yml"


def _publish_text() -> str:
    return _PUBLISH_WORKFLOW.read_text()


def test_resolution_guard_job_exists() -> None:
    _job_lines(_publish_text(), "resolution-guard")  # raises AssertionError if missing


def test_resolution_guard_needs_resolve_and_build() -> None:
    block = "\n".join(_job_lines(_publish_text(), "resolution-guard"))
    needs_line = next(line for line in block.splitlines() if "needs:" in line)
    assert "resolve" in needs_line
    assert "build" in needs_line


def test_publish_testpypi_now_also_needs_resolution_guard() -> None:
    block = "\n".join(_job_lines(_publish_text(), "publish-testpypi"))
    needs_line = next(line for line in block.splitlines() if "needs:" in line)
    assert "resolve" in needs_line
    assert "build" in needs_line
    assert "resolution-guard" in needs_line


def test_publish_job_still_only_needs_resolve_build_publish_testpypi() -> None:
    """The guard sits between `build` and `publish-testpypi`, not after it --
    `publish` keeps its existing `needs:` line unchanged."""
    block = "\n".join(_job_lines(_publish_text(), "publish"))
    needs_line = next(line for line in block.splitlines() if "needs:" in line)
    assert "resolve" in needs_line
    assert "build" in needs_line
    assert "publish-testpypi" in needs_line
    assert "resolution-guard" not in needs_line


def test_resolution_guard_checks_out_the_resolved_sha() -> None:
    block = "\n".join(_job_lines(_publish_text(), "resolution-guard"))
    assert "uses: actions/checkout" in block
    assert "ref: ${{ needs.resolve.outputs.sha }}" in block


def test_resolution_guard_uses_local_setup_uv_action() -> None:
    block = "\n".join(_job_lines(_publish_text(), "resolution-guard"))
    assert "uses: ./.github/actions/setup-uv" in block
    assert "astral-sh/setup-uv@" not in block


def test_resolution_guard_installs_the_release_tooling() -> None:
    """`--only-group` (not `--group`): this job only needs
    workspace_manifests' tomli dependency, not the full default project
    (torch et al.) that a bare `--group release` would additionally pull in.
    """
    block = "\n".join(_job_lines(_publish_text(), "resolution-guard"))
    assert "uv sync --only-group release" in block


def test_resolution_guard_derives_requirements_from_workspace_manifests_cli() -> None:
    block = "\n".join(_job_lines(_publish_text(), "resolution-guard"))
    assert "scripts/workspace_manifests.py requirement-lines" in block
    assert "--repo-root ." in block
    assert '--package "$PACKAGE"' in block
    assert "requirements.in" in block


def test_resolution_guard_is_a_no_op_when_no_dependency_is_declared() -> None:
    block = "\n".join(_job_lines(_publish_text(), "resolution-guard"))
    assert "-s requirements.in" in block
    assert "exit 0" in block


def test_resolution_guard_prints_the_resolved_requirement_set() -> None:
    block = "\n".join(_job_lines(_publish_text(), "resolution-guard"))
    assert "cat requirements.in" in block


def test_resolution_guard_compiles_without_sources_pinned_to_python_310() -> None:
    block = "\n".join(_job_lines(_publish_text(), "resolution-guard"))
    assert "uv pip compile" in block
    assert "--no-sources" in block
    assert "--python-version 3.10" in block


def test_resolution_guard_does_not_install_the_resolved_set() -> None:
    block = "\n".join(_job_lines(_publish_text(), "resolution-guard"))
    assert "--output-file /dev/null" in block
