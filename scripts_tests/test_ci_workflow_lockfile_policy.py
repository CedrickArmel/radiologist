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

"""Behavioral tests for issue #228's lockfile-drift policy.

`.github/actions/setup-uv/action.yml` (issue #225's skeleton) exports
``UV_LOCKED`` with a default of ``"true"``; issue #226 wired every non-test
job onto it while deliberately overriding every call site to ``"false"`` to
keep that slice behaviourless. This issue removes those overrides so the
enforcing default takes effect: ``uv sync``/``uv run`` must fail loudly when
``uv.lock`` disagrees with a manifest, instead of silently re-resolving it on
the runner (``UV_FROZEN`` would accept a stale lock instead -- deliberately
not used, see the module docstring of ``test_ci_workflow_toolchain.py``).

Two more wasteful-resolution paths are closed here:

- ``ci.yml``'s and ``publish.yml``'s ``build`` jobs ran
  ``uv run --with twine twine check dist/*``, which materialises the whole
  project environment (torch included) just to lint distribution metadata.
  Both now run twine in an isolated, ephemeral environment instead.
- ``.github/actions/setup-and-test/action.yml``'s ``make test`` step shells
  out to ``uv run``, which locks and syncs before running -- wasted work,
  since the preceding ``uv sync`` step already installed everything the
  suite needs. That step gains ``UV_NO_SYNC: "true"``.

The single documented exception is `release.yml`'s `cz bump` step, which
legitimately rewrites `uv.lock` as part of the version bump.

These tests parse the workflow/action YAML as text (no ``pyyaml``
dependency), mirroring `scripts_tests/test_ci_workflows_exclude_ray.py` and
`scripts_tests/test_ci_workflow_toolchain.py` (test files in this repo do
not import each other -- the small `_job_lines`-style helpers below are
copied, not shared).
"""

import re
from pathlib import Path
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"
_ACTIONS_DIR = _REPO_ROOT / ".github" / "actions"
_SETUP_AND_TEST_ACTION = _ACTIONS_DIR / "setup-and-test" / "action.yml"

_JOB_HEADER_RE = re.compile(r"^  (\S+):\s*$")
_TOP_LEVEL_KEY_RE = re.compile(r"^(\S+):\s*$")

_WORKFLOW_NAMES = ["ci.yml", "publish.yml", "release.yml", "docs.yml"]


def _read(path: Path) -> str:
    return path.read_text()


def _workflow_paths() -> List[Path]:
    return [_WORKFLOWS_DIR / name for name in _WORKFLOW_NAMES]


def _all_yaml_paths() -> List[Path]:
    return _workflow_paths() + [path for path in _ACTIONS_DIR.rglob("action.yml")]


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


def _top_level_block(workflow_text: str, key: str) -> List[str]:
    """Return the raw lines of a top-level (workflow-level) mapping block."""
    lines = workflow_text.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = _TOP_LEVEL_KEY_RE.match(line)
        if match and match.group(1) == key:
            start = index + 1
            break
    if start is None:
        return []
    end = len(lines)
    for index in range(start, len(lines)):
        if _TOP_LEVEL_KEY_RE.match(lines[index]):
            end = index
            break
    return lines[start:end]


def _job_level_env_block(job_lines: List[str]) -> List[str]:
    """Return the lines of a job's own top-level ``env:`` block, if any.

    A job-level ``env:`` is indented one level deeper than the job name but
    shallower than a step; steps start with ``- `` under a ``steps:`` key, so
    the job-level ``env:`` (if present) always appears before any ``steps:``
    line.
    """
    try:
        steps_index = next(
            index for index, line in enumerate(job_lines) if line.strip() == "steps:"
        )
    except StopIteration:
        steps_index = len(job_lines)
    preamble = job_lines[:steps_index]
    for index, line in enumerate(preamble):
        if line.strip() == "env:":
            block = []
            for candidate in preamble[index + 1 :]:
                if candidate.startswith("    ") and not candidate.startswith("      "):
                    block.append(candidate)
                else:
                    break
            return block
    return []


def _all_job_names(workflow_text: str) -> List[str]:
    lines = workflow_text.splitlines()
    job_section = _top_level_block(workflow_text, "jobs")
    names = []
    for line in job_section:
        match = _JOB_HEADER_RE.match(line)
        if match:
            names.append(match.group(1))
    return names or [m.group(1) for m in map(_JOB_HEADER_RE.match, lines) if m]


def test_no_workflow_declares_uv_locked_at_workflow_level() -> None:
    """A workflow-level `env:` would override every job's own `$GITHUB_ENV`.

    `UV_LOCKED` has exactly two legitimate owners: `setup-uv`'s input
    (job-wide, via `$GITHUB_ENV`) and a step-level `env:` override. A
    workflow-level `env:` block sits above job scope and would silently
    disable any step-level opt-out underneath it.
    """
    for path in _workflow_paths():
        text = _read(path)
        top_env = _top_level_block(text, "env")
        block = "\n".join(top_env)
        assert "UV_LOCKED" not in block, f"{path.name} sets UV_LOCKED at workflow level"


def test_no_job_declares_uv_locked_at_job_level() -> None:
    """A job-level `env:` would override `setup-uv`'s `$GITHUB_ENV` write.

    Only a step-level `env:` may declare `UV_LOCKED` -- see
    `test_exactly_one_step_opts_out_of_the_locked_policy`.
    """
    for path in _workflow_paths():
        text = _read(path)
        for job_name in _all_job_names(text):
            job_env = _job_level_env_block(_job_lines(text, job_name))
            block = "\n".join(job_env)
            assert (
                "UV_LOCKED" not in block
            ), f"{path.name}:{job_name} sets UV_LOCKED at job level"


def test_uv_frozen_is_never_used_as_the_drift_gate() -> None:
    """UV_FROZEN/--frozen silently accepts a stale lockfile; only UV_LOCKED errors.

    A prose comment *naming* UV_FROZEN to explain why it is deliberately not
    used is fine and expected; what must never appear is an actual
    assignment/flag that would activate it.
    """
    assignment_re = re.compile(r'UV_FROZEN\s*[:=]\s*"?(true|1)"?', re.IGNORECASE)
    for path in _all_yaml_paths():
        for line in _read(path).splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue  # prose explaining why --frozen/UV_FROZEN is avoided is fine
            assert not assignment_re.search(line), f"{path} sets UV_FROZEN"
            assert "--frozen" not in line, f"{path} uses --frozen: {line!r}"


def test_no_call_site_still_passes_locked_false_to_setup_uv() -> None:
    """Issue #226 left `locked: \"false\"` at every call site for #228 to flip."""
    for path in _workflow_paths():
        text = _read(path)
        assert (
            'locked: "false"' not in text
        ), f"{path.name} still overrides setup-uv's locked default to false"


def test_exactly_one_step_opts_out_of_the_locked_policy() -> None:
    hits = []
    for path in _workflow_paths():
        text = _read(path)
        for line in text.splitlines():
            if re.search(r'UV_LOCKED:\s*"false"', line):
                hits.append((path.name, line.strip()))
    assert len(hits) == 1, f'expected exactly one UV_LOCKED: "false" step, got {hits!r}'


def test_the_opt_out_is_on_release_yml_cz_bump_step_and_is_documented() -> None:
    text = _read(_WORKFLOWS_DIR / "release.yml")
    lines = text.splitlines()
    opt_out_index = next(
        index for index, line in enumerate(lines) if 'UV_LOCKED: "false"' in line
    )
    # Walk up to the nearest `- name:` step header to scope the search.
    step_start = opt_out_index
    while step_start > 0 and not lines[step_start].strip().startswith("- name:"):
        step_start -= 1
    assert "cz bump" in "\n".join(lines[opt_out_index : opt_out_index + 15])
    # A comment inside the step (above `env:`/`run:`) must explain the
    # exemption -- it need not sit above the `- name:` header itself.
    comment_lines = [
        line for line in lines[step_start:opt_out_index] if line.strip().startswith("#")
    ]
    comment_block = "\n".join(comment_lines).lower()
    assert comment_block.strip(), "no comment documents the UV_LOCKED opt-out"
    assert "lock" in comment_block


def test_no_workflow_runs_twine_through_the_project_environment() -> None:
    for path in [_WORKFLOWS_DIR / "ci.yml", _WORKFLOWS_DIR / "publish.yml"]:
        text = _read(path)
        assert (
            "uv run --with twine" not in text
        ), f"{path.name} still uses uv run --with twine"


def test_twine_check_runs_isolated_in_ci_and_publish_build_jobs() -> None:
    for workflow in ["ci.yml", "publish.yml"]:
        block = "\n".join(_job_lines(_read(_WORKFLOWS_DIR / workflow), "build"))
        assert re.search(
            r"uvx twine check dist/\*", block
        ), f"{workflow}'s build job does not run twine via an isolated uvx invocation"


def test_setup_and_test_sync_step_enforces_the_locked_policy() -> None:
    text = _read(_SETUP_AND_TEST_ACTION)
    lines = text.splitlines()
    sync_index = next(
        index
        for index, line in enumerate(lines)
        if "uv sync" in line and "--all-extras" in line
    )
    following = "\n".join(lines[sync_index : sync_index + 10])
    assert 'UV_LOCKED: "true"' in following


def test_setup_and_test_test_step_skips_the_redundant_sync() -> None:
    text = _read(_SETUP_AND_TEST_ACTION)
    lines = text.splitlines()
    test_index = next(
        index
        for index, line in enumerate(lines)
        if line.strip().startswith("- run:") and "make test" in line
    )
    following = "\n".join(lines[test_index : test_index + 10])
    assert 'UV_NO_SYNC: "true"' in following


def test_setup_and_test_install_command_string_is_unchanged() -> None:
    """The install command line itself is untouched -- only env: is added below it."""
    text = _read(_SETUP_AND_TEST_ACTION)
    assert (
        "- run: uv sync --active --all-groups --all-packages --all-extras "
        "--no-extra ray" in text
    )


def test_setup_and_test_test_command_string_is_unchanged() -> None:
    text = _read(_SETUP_AND_TEST_ACTION)
    assert '- run: make test PYTEST_FLAGS="${{ inputs.pytest-flags }}"' in text


def test_ray_exclusion_comment_still_sits_directly_above_the_install_step() -> None:
    """Adding env: below the run: line must not disturb the #188 comment block."""
    lines = _read(_SETUP_AND_TEST_ACTION).splitlines()
    sync_index = next(
        index
        for index, line in enumerate(lines)
        if "uv sync" in line and "--all-extras" in line
    )
    start = sync_index
    while start > 0 and lines[start - 1].strip().startswith("#"):
        start -= 1
    comment_block = "\n".join(lines[start:sync_index]).lower()
    assert "ray" in comment_block
    assert "188" in comment_block


def test_no_no_sync_step_lacks_a_preceding_explicit_sync_in_the_same_job() -> None:
    """UV_NO_SYNC must only appear where an explicit `uv sync` already ran."""
    for path in _all_yaml_paths():
        text = _read(path)
        if "release.yml" in str(path):
            continue
        lines = text.splitlines()
        no_sync_lines = [i for i, line in enumerate(lines) if "UV_NO_SYNC" in line]
        for index in no_sync_lines:
            preceding = "\n".join(lines[:index])
            assert (
                "uv sync" in preceding
            ), f"{path}: UV_NO_SYNC used without a preceding explicit uv sync"
