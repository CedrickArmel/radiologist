---
name: project-provenance-ci-dedupe-167
description: Issue #167 (dedupe ci.yml/publish.yml test gate) required an explicit user override of the issue's own "no test may be changed" AC
metadata:
  type: project
---

Issue #167 was a parked refactor ("dedupe the duplicated test-gate steps between
`.github/workflows/ci.yml` and `.github/workflows/publish.yml` into a composite
action") with two self-declared preconditions for starting: a third consumer of
the shared steps appears, or the two copies drift. Neither had happened as of
2026-09-05/06 — the just-merged epic PR #222 had in fact kept the duplication
byte-identical on purpose. Initial recommendation was to close as won't-do.

The user (via the coordinator) explicitly overrode that recommendation and
asked to proceed anyway. During implementation a second, concrete blocker
surfaced: `scripts_tests/test_ci_workflows_exclude_ray.py` (added by sibling
issue #216) asserts directly on the inline `uv sync ... --no-extra ray` line
inside each workflow's `test` job body — extracting that line into
`.github/actions/setup-and-test/action.yml` necessarily breaks 5 of its 8
assertions, which directly contradicts #167's own AC "All existing tests pass
without modification ... if one must change, the scope is wrong."

The user again explicitly chose to override this AC: do the full extraction
(composite action carries checkout + setup-python + setup-uv + uv sync +
make test, parameterized by `ref` and `pytest-flags` to preserve ci.yml's
Codecov-only `--cov-report=xml` vs publish.yml's plain `-q`), and relocate
the test file's assertions to check the composite action file instead of the
inline job body — preserving all 8 original behavioral assertions, just
pointed at the new location.

**Why:** the issue's own preconditions and the "no test changes" AC exist
precisely to avoid premature abstraction with hidden costs; this session is a
worked example of that hidden cost showing up (a sibling issue's regression
test had silently made the duplication load-bearing). The user's calls here
were explicit, informed overrides at each blocking point, not oversights —
worth remembering that this project's owner is willing to spend that cost
once shown it concretely, rather than always defaulting to won't-do.

**How to apply:** when a "parked, don't start until X" issue is explicitly
unparked by the user, still surface every concrete blocker you find *before*
touching it (including ones invisible from reading the issue text alone,
like a sibling issue's test suite becoming coupled to the duplication) —
don't silently modify tests or silently leave the refactor half-done. Get an
explicit call each time, and document the override in the commit message.
See [[feedback_optional_refactor_wont_do_bar]] for the general pattern this
is an exception to.
