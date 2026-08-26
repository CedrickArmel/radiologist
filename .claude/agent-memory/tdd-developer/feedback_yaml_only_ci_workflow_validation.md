---
name: yaml-only-ci-workflow-validation
description: How to validate a pure GitHub Actions workflow (no unit tests) before committing
metadata:
  type: feedback
---

A workflow-only issue (e.g. adding `.github/workflows/ci.yml`) has no unit tests to drive
red→green→refactor against. Treat it as the TDD skill's documented exception: validate
empirically instead —

1. `python3 -c "import yaml; yaml.safe_load(open(path))"` for basic syntax.
2. Download `actionlint` (`curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash`)
   and run it against the new file — it catches expression-context errors, unknown action
   inputs, and shell issues `yaml.safe_load` can't. Delete the downloaded binary/script
   before committing; it is not a repo artifact.
3. **Dry-run every shell step the workflow invokes, locally, exactly as written** — e.g. if
   the workflow does `make build-all` then `uv run --with twine twine check dist/*`, run
   those two commands yourself in a real (worktree-scoped pyenv) venv and confirm they
   succeed from a clean tree. This is the only way to catch a stale Makefile target or a
   typo'd twine invocation before CI actually runs it.
4. A pre-existing test/mypy failure unrelated to your diff is out of scope for a CI-only
   issue — confirm via `git stash -u` that your new/untracked files are the only diff, then
   note the pre-existing failure in your report rather than fixing it.

Why: GitHub Actions can't be executed locally, so "validate" for this issue-type means
"prove every individual step succeeds outside the workflow harness," not "watch a red
test." See [[shared/MEMORY.md]] for the general TDD skill exception for skeleton issues,
which this generalizes to workflow-only issues.

How to apply: any future issue that adds/edits `.github/workflows/*.yml` (e.g. #165's
`release.yml`/`publish.yml`, which the epic says will duplicate this issue's test steps).
