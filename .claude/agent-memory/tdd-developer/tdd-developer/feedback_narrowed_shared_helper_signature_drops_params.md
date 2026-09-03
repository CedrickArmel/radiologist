---
name: narrowed-shared-helper-signature-drops-params
description: an issue's literal interface-contract code snippet for a shared loading/dispatch helper can omit optional kwargs (mean/std/input_shape-style passthrough) that landed in the repo after the snippet was drafted — grep existing tests for the old call sites before deleting the params
metadata:
  type: feedback
---

In radiologist#143 (CLI rewire onto `verbs.load_predictor`), the issue's interface contract
showed the exact call `verbs.load_predictor(verb, model, run_id, tags, groups, metric,
local_dir)` — 7 positional args, no `mean`/`std`/`input_shape`. But `verbs.py` (#142) had been
drafted before a separate, already-merged bugfix (#139/commits `e30e572`/`9d9ea35`) added
`mean`/`std`/`input_shape` normalization-override support to `Classifier.from_path` /
`from_selector` and threaded it through the CLI's old `_load_predictor` /
`_load_uncertainty_predictor` helpers. `tests/test_cli.py` already had
`TestPredictCommandNormalizationFlags` etc. asserting `--mean --std --input-shape` change
CLI output — those tests were not mentioned in #143's "Test updates" list at all, because the
issue author's snippet simply predated that feature landing.

**Why:** literal interface-contract snippets in an issue are drafted at spec time and can miss
capabilities that merged into `main`/the epic branch afterward. Deleting the params to match
the snippet exactly would silently drop working normalization-override support and break
already-passing tests — "pytest green" as an AC always wins over a narrower literal snippet.

**How to apply:** before implementing a "rewire onto shared helper X" issue, `grep -rn` the
old call sites the issue is replacing for **all** their kwargs (not just the ones named in the
issue body), and check whether the shared helper's current signature already supports them. If
not, extend the shared helper with the same optional kwargs (default `None`), forwarded
verbatim to the underlying constructor — this keeps the literal positional-arg contract intact
(callers using only the 7 named args are unaffected) while not regressing capability. Related:
[[feedback_shared_base_class_seam_in_decomposition_epics]] (implement the seam your own
GREEN-real bar needs) and [[feedback_issue_test_scope_incomplete]] (grep the whole test suite,
not just the issue's named line ranges).
