---
name: project-publish-workflow-resolve-pattern
description: how publish.yml's resolve job gets a trustworthy checkout+sha before the sha is "computed", and why tomli must be an explicit release-group dependency
metadata:
  type: project
---

Epic "Publish the radiologist mono-repo to PyPI" (milestone #14), issue #165
(`.github/workflows/publish.yml`). Two non-obvious points worth remembering
if this workflow is revisited:

1. **The "resolve" job's sha is known before any checkout, not computed by
   one.** On a `pull_request: types: [closed]` event, `merge_commit_sha` is
   already present in the event payload the instant the workflow starts —
   it does not require checking anything out first. So `resolve` checks out
   `github.event.pull_request.merge_commit_sha || github.sha` **directly**
   as its first step, then reads `[project].version` straight out of that
   checked-out working tree. This avoids the tempting-but-wrong pattern of
   checking out the default (PR merge) ref first and trying to `git show
   $SHA:path` afterward — the default ref for a `pull_request: closed` event
   points at a merge ref that stops existing once the PR is merged.

2. **`scripts/release_bump.py` needs `tomllib`/`tomli` to read
   `[project].version`, but this repo pins Python 3.10 (no stdlib
   `tomllib`).** `tomli` was already present transitively (via
   `commitizen`), which is fragile — a future commitizen version could drop
   it. Added `tomli>=2.0.1; python_version < '3.11'` as an explicit member of
   the root `pyproject.toml`'s `release` dependency-group so `uv sync
   --group release` guarantees it, with a
   `try: import tomllib except ModuleNotFoundError: import tomli as tomllib`
   shim in the module so nothing breaks if the repo ever bumps past 3.11.

3. **`release_bump.py` gained an inverse of issue #164's `release_branch_name`**:
   `parse_release_branch_name` splits `release/<package>-v<version>` on the
   *last* `-v` (so hyphenated names like `radiologist-core` parse right),
   plus `environment_name` (`pypi-<package>`) and `release_tag` (bare
   version for the root `radiologist` package, `<version>-<package>` for
   every workspace member — this matches each member's `[tool.commitizen]
   tag_format` verified by grep, not guessed). All four are pure functions,
   TDD'd in `scripts_tests/test_release_bump.py`, and exposed as `uv run
   python scripts/release_bump.py <subcommand>` CLI calls so the workflow
   YAML never re-derives the parsing/format logic in bash.

See also [[feedback_cz_uv_provider_cwd_relative_lockfile]] for the sibling
gotcha about `cz bump`'s cwd-relative lockfile resolution from issue #164.
