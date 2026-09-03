---
name: docstring-lint-cutover-scope
description: When flipping a suppressed lint rule to hard-failing repo-wide, the "clean" claim from a prior verification run is only as trustworthy as the config it was measured under — re-verify after the actual flip, and expect the blast radius to be wider than the stated target path
metadata:
  type: feedback
---

On issue #138 (final "flip" phase of a docs epic), the orchestrator's
prerequisite claim was "`flake8 --select=D radiologist-*/src` is already
clean" — but that check had been run while `.flake8`'s `extend-ignore`
still contained `D1` (the code family for D100/D103/D104, missing
module/function/package docstrings). `--select=D` does not override
`extend-ignore` in the same invocation, so the ignored D1 codes never
surfaced as findings regardless of `--select`. The "zero findings" was
true only under the still-suppressed config — not evidence the flip
itself would be clean.

**Why this matters generically:** any epic phase that "just removes a
suppression flag" is not config-only work — it's the first time the
underlying checks actually run. Don't trust a sibling agent's or
orchestrator's "verified clean" claim about a rule that was suppressed
when they measured it. Re-run the check in the *exact* config state
you are about to ship.

**Second-order surprise:** a repo-wide pre-commit flake8 hook (no path
filter) applies the newly-enforced docstring convention to `tests/`
directories too, even though the issue's stated AC only scoped
`flake8 --select=D` to `radiologist-*/src`. Writing Google docstrings
into every test function across 5 packages would have been enormous
scope creep for a "flip" issue. The correct fix was a `per-file-ignores
= */tests/*: D` entry in `.flake8` — tests are not the public API
surface the docstring convention (and mkdocstrings site) targets, so
exempting them from D-codes while keeping full enforcement on `src/`
satisfies both the letter (`src` clean) and the practical requirement
(`pre-commit run --all-files` passes) without doing out-of-scope work.
A root-level `conftest.py` outside any `tests/` directory still needed
its own one-line module docstring since the glob didn't match it.

**How to apply:** before trusting "prerequisite already verified
clean" language in an epic/issue spec, re-run the literal command
yourself against the current file state you're about to commit — not
the state described. When a suppression removal surfaces gaps the
epic didn't anticipate, close real src/ gaps (that's this issue's
actual job) but push back on scope via a per-file-ignore, not by
writing test docstrings, when the AC's own wording already scoped
"clean" to `src/`.
