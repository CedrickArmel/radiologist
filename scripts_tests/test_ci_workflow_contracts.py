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

"""Behavioral tests for this repository's CI/CD workflow invariants.

Issue #235 folded `.github/actions/setup-and-test/action.yml` onto
`.github/actions/setup-uv/action.yml` for its toolchain-setup portion (no
duplicated `actions/setup-python`/`astral-sh/setup-uv` pins remain outside
`setup-uv`). As its own scope, it also consolidates three test files that had
each accreted around one epic issue into this single file, organised by
*invariant* rather than by the issue that introduced it:

- Ray-exclusion contract (previously `test_ci_workflows_exclude_ray.py`,
  issue #216, later relocated onto the composite action by issue #167).
- Toolchain adoption of `setup-uv` (previously `test_ci_workflow_toolchain.py`,
  issue #226).
- Lockfile-drift policy (previously `test_ci_workflow_lockfile_policy.py`,
  issue #228).

Per issue #235's own acceptance criteria, every assertion below is an
unmodified relocation of an existing test -- the one sanctioned exception
inherited from issue #167's own precedent (see that issue's module docstring,
preserved in the Ray-exclusion section) is moving what a test asserts *about*,
never changing what it asserts. Only the tiny per-file helper functions
(`_job_lines` and friends) were deduplicated, since those are test plumbing,
not behavioral assertions.

These tests parse the workflow/action YAML as text (no ``pyyaml`` dependency)
so they stay green on a checkout where no extra at all is installed.
"""

import re
from pathlib import Path
from typing import List, Tuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS_DIR = _REPO_ROOT / ".github" / "workflows"
_ACTIONS_DIR = _REPO_ROOT / ".github" / "actions"
_SETUP_AND_TEST_ACTION = _ACTIONS_DIR / "setup-and-test" / "action.yml"
_SETUP_UV_ACTION = _ACTIONS_DIR / "setup-uv" / "action.yml"
# Alias kept for parity with the pre-consolidation Ray-exclusion test file,
# which named the same path differently.
_COMPOSITE_ACTION = _SETUP_AND_TEST_ACTION

_JOB_HEADER_RE = re.compile(r"^  (\S+):\s*$")
_TOP_LEVEL_KEY_RE = re.compile(r"^(\S+):\s*$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

_WORKFLOW_NAMES = ["ci.yml", "publish.yml", "release.yml", "docs.yml"]

# Every non-test job that hand-rolled the toolchain pair before issue #226.
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


def _read(path: Path) -> str:
    return path.read_text()


def _read_workflow(name: str) -> str:
    return (_WORKFLOWS_DIR / name).read_text()


def _read_composite_action() -> str:
    return _COMPOSITE_ACTION.read_text()


def _composite_action_lines() -> List[str]:
    return _read_composite_action().splitlines()


def _workflow_texts() -> List[str]:
    return [path.read_text() for path in _WORKFLOWS_DIR.glob("*.yml")]


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


def _preceding_comment_block(job_lines: List[str], step_index: int) -> str:
    """Return the contiguous run of ``#``-comment lines directly above a step."""
    start = step_index
    while start > 0 and job_lines[start - 1].strip().startswith("#"):
        start -= 1
    return "\n".join(job_lines[start:step_index]).lower()


# ---------------------------------------------------------------------------
# Invariant: Ray is excluded from the two `test` jobs' installs (issue #216,
# relocated onto the composite action by issue #167).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workflow", ["ci.yml", "publish.yml"])
def test_test_job_delegates_to_the_shared_setup_and_test_action(
    workflow: str,
) -> None:
    """Each workflow's `test` job installs/tests via the composite action.

    The install command itself no longer lives inline in the job body -- what
    each job body must still show is that it calls the one shared action
    rather than re-declaring its own steps.
    """
    lines = _job_lines(_read_workflow(workflow), "test")
    block = "\n".join(lines)
    assert "uses: ./.github/actions/setup-and-test" in block
    assert "uv sync" not in block, (
        f"{workflow}: the install command should live only in the composite "
        "action, not duplicated inline in the job body"
    )


@pytest.mark.parametrize("workflow", ["ci.yml", "publish.yml"])
def test_test_job_checks_out_before_calling_the_local_composite_action(
    workflow: str,
) -> None:
    """A local `uses: ./...` action can't be resolved before a checkout.

    GitHub loads a same-repo composite action off the runner's local
    filesystem, which is empty until something checks the repository out --
    the composite action's own internal checkout runs too late to bootstrap
    itself. Each `test` job must therefore run `actions/checkout` before
    invoking `setup-and-test`.
    """
    lines = _job_lines(_read_workflow(workflow), "test")
    composite_index = next(
        index for index, line in enumerate(lines) if "uses: ./.github/actions" in line
    )
    preceding = "\n".join(lines[:composite_index])
    assert "uses: actions/checkout" in preceding, (
        f"{workflow}: the test job must check out the repo before invoking "
        "the local setup-and-test composite action"
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


# ---------------------------------------------------------------------------
# Invariant: every non-test job is on the local `setup-uv` composite action,
# not a hand-rolled `actions/setup-python` + `astral-sh/setup-uv` pair
# (issue #226).
# ---------------------------------------------------------------------------


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
def test_target_job_no_longer_overrides_the_locked_default(
    workflow: str, job: str
) -> None:
    """Issue #228 flipped every `locked: "false"` override -- see the
    lockfile-drift-policy section below for the full contract this now
    leaves in force (`setup-uv`'s enforcing `"true"` default applies at
    every one of these call sites, except release.yml's single documented
    `cz bump` step-level override)."""
    block = "\n".join(_job_lines(_read_workflow(workflow), job))
    assert 'locked: "false"' not in block


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
    """`setup-and-test` still owns checkout + sync + `make test`.

    Written against issue #226, which deliberately did not fold
    `setup-and-test` onto `setup-uv` yet. Issue #235 performs that fold for
    the toolchain-setup portion only -- checkout, the `uv sync` install step
    and `make test` remain `setup-and-test`'s own responsibility, so this
    assertion still holds unmodified.
    """
    text = _read(_SETUP_AND_TEST_ACTION)
    assert "actions/checkout" in text
    assert "uv sync" in text
    assert "make test" in text


# ---------------------------------------------------------------------------
# Invariant: `UV_LOCKED` lockfile-drift enforcement (issue #228), with the
# single documented exception on release.yml's `cz bump` step.
# ---------------------------------------------------------------------------


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


def test_release_only_group_sync_never_installs_the_full_workspace() -> None:
    """`--group release` *adds* to the default (project) sync; only
    `--only-group release` skips installing torch/lightning/etc. for a job
    that just needs commitizen/tomli."""
    for path in _workflow_paths():
        text = _read(path)
        assert (
            "uv sync --group release" not in text
        ), f"{path.name} still uses --group release"


def _step_blocks(job_lines: List[str]) -> List[List[str]]:
    """Split a job's lines into per-step blocks on 6-space-indented `- `."""
    step_re = re.compile(r"^      - ")
    blocks: List[List[str]] = []
    for line in job_lines:
        if step_re.match(line):
            blocks.append([line])
        elif blocks:
            blocks[-1].append(line)
    return blocks


@pytest.mark.parametrize(
    "workflow,job",
    [
        ("publish.yml", "resolve"),
        ("publish.yml", "resolution-guard"),
        ("publish.yml", "tag"),
        ("release.yml", "bump"),
    ],
)
def test_uv_run_steps_after_the_only_group_sync_skip_the_redundant_sync(
    workflow: str, job: str
) -> None:
    """Every `uv run` step in these jobs only needs what the preceding
    `--only-group release` sync already installed. The one documented
    exception is release.yml's cz-bump step, which legitimately rewrites
    `uv.lock` and is exempted from the drift gate by its own `UV_LOCKED:
    "false"` -- covered by test_the_opt_out_is_on_release_yml_cz_bump_step_
    and_is_documented, not here.
    """
    no_sync_re = re.compile(r'^\s*UV_NO_SYNC:\s*"true"\s*$')
    job_lines = _job_lines(_read_workflow(workflow), job)
    for block in _step_blocks(job_lines):
        code_lines = [line for line in block if not line.strip().startswith("#")]
        code_text = "\n".join(code_lines)
        if "uv run" not in code_text:
            continue
        if "cz bump" in code_text:
            continue
        assert any(no_sync_re.match(line) for line in code_lines), (
            f'{workflow}:{job}: a uv run step lacks a real UV_NO_SYNC: "true" '
            f"env line and will re-materialise the full default project:\n"
            f"{code_text}"
        )
