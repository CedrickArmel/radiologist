---
name: skeleton-issue-repoints-old-tests
description: When a skeleton issue removes a class from a package's public __init__, repoint (don't leave broken) existing behavioral tests that imported it via the package
metadata:
  type: feedback
---

A "skeleton" issue that decomposes a monolith (e.g. `Predictor` ->
`BasePredictor`/`Classifier`/`Explainer`/`MCDropoutPredictor`) will often say
existing behavioral test files "still import the old class and will be
migrated by later slice issues" — implying it's fine to leave them broken.
In practice this conflicts with the issue's own "pytest green" acceptance
criterion: if the monolith's class is dropped from the package's
`__init__.py` `__all__`/exports, every test that did
`from package import OldClass` fails at **collection**, which aborts the
whole pytest run (not just those tests) and is a much worse outcome than a
handful of assertion failures.

**Why:** the old monolith module itself (e.g. `predictor.py`) is
deliberately left untouched and fully functional during the skeleton phase
— only the package-level export is what changed. The fix belongs at the
import site, not in the monolith or the new stubs.

**How to apply:** grep the test suite for `from <package> import
<ClassBeingRemoved>` (including inside `conftest.py` fixtures) and repoint
those specific import lines to the internal module that still defines the
class (`from <package>.<old_module> import <ClassBeingRemoved>`), leaving
every assertion and fixture body untouched. Do this for every name that
travels with the removed class on the same import line (e.g. `Prediction`,
`Explanation`, `UncertaintyResult` returned by the old class's methods) —
mixing the new frozen dataclasses from the skeleton's `models.py` with
instances produced by the old class breaks `isinstance` checks, since
they're different class objects even with identical fields.

If the issue also changes a CLI's command surface (removing/renaming
subcommands), the old CLI test file needs a **rewrite** to shape-only
assertions (command names present/absent, extra-guard behavior) rather than
behavioral assertions — the same pattern as the rewritten
`test_public_api.py`. This is not "adding a new behavioral test"; it's
keeping the existing shape-level test in sync with the now-different CLI
surface.

See also [[shared/feedback_tdd]] for the general skeleton-issue exception
(stub bodies get no tests) — this memory is about the surrounding test
suite, not the stubs themselves.
