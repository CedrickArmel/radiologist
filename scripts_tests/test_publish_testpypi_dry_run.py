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

"""Behavioral tests for issue #230's TestPyPI dry-run publish job.

`.github/workflows/publish.yml`'s `publish` job uploads straight to PyPI with
no rehearsal. Since a PyPI filename can never be reused, a metadata defect
that `twine check` misses burns that version permanently. This issue adds a
`publish-testpypi` job that consumes the same `build` artifact, uploads it to
TestPyPI first, and gates the real `publish` job behind its success.

These tests parse the workflow YAML as text (no `pyyaml` dependency),
mirroring `scripts_tests/test_ci_workflows_exclude_ray.py` and
`scripts_tests/test_ci_workflow_toolchain.py`.
"""

import re
from pathlib import Path
from typing import List

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[import-not-found, no-redef]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PUBLISH_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "publish.yml"
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

_JOB_HEADER_RE = re.compile(r"^  (\S+):\s*$")


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


def test_publish_testpypi_job_exists() -> None:
    _job_lines(_publish_text(), "publish-testpypi")  # raises AssertionError if missing


def test_publish_testpypi_needs_resolve_and_build() -> None:
    block = "\n".join(_job_lines(_publish_text(), "publish-testpypi"))
    needs_line = next(line for line in block.splitlines() if "needs:" in line)
    assert "resolve" in needs_line
    assert "build" in needs_line


def test_publish_job_now_also_needs_publish_testpypi() -> None:
    block = "\n".join(_job_lines(_publish_text(), "publish"))
    needs_line = next(line for line in block.splitlines() if "needs:" in line)
    assert "resolve" in needs_line
    assert "build" in needs_line
    assert "publish-testpypi" in needs_line


def test_publish_testpypi_downloads_the_same_build_artifact_as_publish() -> None:
    testpypi_block = "\n".join(_job_lines(_publish_text(), "publish-testpypi"))
    publish_block = "\n".join(_job_lines(_publish_text(), "publish"))
    artifact_name = (
        "dist-${{ needs.resolve.outputs.package }}-"
        "${{ needs.resolve.outputs.version }}"
    )
    assert artifact_name in testpypi_block
    assert artifact_name in publish_block
    assert "uses: actions/download-artifact" in testpypi_block


def test_publish_testpypi_uses_a_distinct_oidc_environment() -> None:
    block = "\n".join(_job_lines(_publish_text(), "publish-testpypi"))
    assert "testpypi-${{ needs.resolve.outputs.package }}" in block
    assert "id-token: write" in block
    assert "contents: read" in block


def test_publish_testpypi_environment_is_distinct_from_the_pypi_one() -> None:
    testpypi_block = "\n".join(_job_lines(_publish_text(), "publish-testpypi"))
    publish_block = "\n".join(_job_lines(_publish_text(), "publish"))
    testpypi_env = next(
        line.strip() for line in testpypi_block.splitlines() if "name:" in line
    )
    publish_env = next(
        line.strip() for line in publish_block.splitlines() if "name:" in line
    )
    assert testpypi_env != publish_env


def test_publish_testpypi_uses_local_setup_uv_action_without_python_setup() -> None:
    block = "\n".join(_job_lines(_publish_text(), "publish-testpypi"))
    assert "uses: ./.github/actions/setup-uv" in block
    assert 'setup-python: "false"' in block
    assert "actions/setup-python@" not in block
    assert "astral-sh/setup-uv@" not in block


def test_publish_testpypi_checks_out_before_the_local_composite_action() -> None:
    lines = _job_lines(_publish_text(), "publish-testpypi")
    composite_index = next(
        index
        for index, line in enumerate(lines)
        if "uses: ./.github/actions/setup-uv" in line
    )
    preceding = "\n".join(lines[:composite_index])
    assert "uses: actions/checkout" in preceding


def test_publish_testpypi_uploads_to_testpypi_index_with_trusted_publishing_always() -> (
    None
):
    block = "\n".join(_job_lines(_publish_text(), "publish-testpypi"))
    assert "uv publish" in block
    assert "--index testpypi" in block
    assert "--trusted-publishing always" in block


def test_publish_testpypi_upload_is_idempotent_via_check_url() -> None:
    """A re-run after a later job failed must not re-burn a version number."""
    block = "\n".join(_job_lines(_publish_text(), "publish-testpypi"))
    assert "--check-url" in block
    assert "test.pypi.org/simple" in block


def test_root_manifest_declares_an_explicit_testpypi_index() -> None:
    with _PYPROJECT.open("rb") as handle:
        data = tomllib.load(handle)
    indexes = data["tool"]["uv"]["index"]
    testpypi = next(index for index in indexes if index["name"] == "testpypi")
    assert testpypi["url"] == "https://test.pypi.org/simple/"
    assert testpypi["publish-url"] == "https://test.pypi.org/legacy/"
    assert testpypi["explicit"] is True
