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

"""Behavioral tests for issue #234's build-provenance attestation step.

`.github/workflows/publish.yml`'s `build` job produces the wheel/sdist that
`publish-testpypi` and `publish` later upload, but nothing records a
verifiable statement of what commit and workflow produced those bytes. This
adds a SLSA-style provenance attestation (`actions/attest-build-provenance`)
generated for the built artifacts before they are uploaded for downstream
jobs, so no job can consume an unattested artifact.

These tests parse the workflow YAML as text (no `pyyaml` dependency),
mirroring `scripts_tests/test_publish_testpypi_dry_run.py`.
"""

import re
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PUBLISH_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "publish.yml"

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


def test_build_job_grants_attestations_and_id_token_write() -> None:
    block = "\n".join(_job_lines(_publish_text(), "build"))
    assert "attestations: write" in block
    assert "id-token: write" in block
    assert "contents: read" in block


def test_build_job_uses_attest_build_provenance_pinned_to_a_sha() -> None:
    block = "\n".join(_job_lines(_publish_text(), "build"))
    match = re.search(
        r"uses: actions/attest-build-provenance@([0-9a-f]{40})\s*#\s*(v[\d.]+)",
        block,
    )
    assert match is not None, "attest-build-provenance step missing or not SHA-pinned"


def test_build_job_attests_the_dist_subject_path() -> None:
    block = "\n".join(_job_lines(_publish_text(), "build"))
    attest_index = next(
        index
        for index, line in enumerate(block.splitlines())
        if "uses: actions/attest-build-provenance" in line
    )
    following = "\n".join(block.splitlines()[attest_index : attest_index + 4])
    assert "subject-path: dist/*" in following


def test_build_job_attests_before_uploading_the_dist_artifact() -> None:
    lines = _job_lines(_publish_text(), "build")
    attest_index = next(
        index
        for index, line in enumerate(lines)
        if "uses: actions/attest-build-provenance" in line
    )
    upload_index = next(
        index
        for index, line in enumerate(lines)
        if "uses: actions/upload-artifact" in line
    )
    assert attest_index < upload_index


def test_id_token_write_is_granted_to_no_job_beyond_build_and_publish_jobs() -> None:
    text = _publish_text()
    lines = text.splitlines()
    jobs_start = next(
        index for index, line in enumerate(lines) if line.strip() == "jobs:"
    )
    job_names = [
        match.group(1)
        for line in lines[jobs_start:]
        for match in [_JOB_HEADER_RE.match(line)]
        if match
    ]
    granting_jobs = [
        job
        for job in job_names
        if "id-token: write" in "\n".join(_job_lines(text, job))
    ]
    assert set(granting_jobs) == {"build", "publish-testpypi", "publish"}
