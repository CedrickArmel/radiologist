---
name: stubbed-shared-decorator-blocks-sibling-slice
description: a shared error-handling decorator stubbed to raise NotImplementedError breaks eagerly at decoration time, not call time — a sibling slice can't use it as-is even when told to "route through" it
metadata:
  type: feedback
---

In the unified-CLI epic (#15), issue #172 (registry group) was told to route
error handling through `radiologist.cli.errors.exit_on_error`, but that
decorator was still stubbed `raise NotImplementedError` (owned by a parallel
sibling issue, #176) and the task instructions explicitly said not to touch
`errors.py`/`main.py`/`optional.py`. Using `@exit_on_error` as-is would fail
at *decoration* time (`exit_on_error(func)` itself raises), not just at call
time — so it can't be imported/applied at all, let alone mocked away.

**Resolution used:** implemented a module-local `_exit_on_error` inside
`groups/registry.py` that mirrors the intended contract, built on the
already-real, already-implemented pieces of the shared seam
(`radiologist.utils.cli.exit_code_for`, `EXIT_NOT_FOUND`/`EXIT_ERROR`) rather
than the still-stubbed decorator wrapper itself. This satisfies the issue's
own acceptance criteria (exit 1 with `Error:` on stderr, exit 2 for a missing
artifact) without touching files owned by a parallel issue. Left a docstring
note that it can be swapped for the shared decorator once #176 lands.

**Why:** "route error handling through the shared decorator" in an epic
context block is aspirational/architectural intent, not a hard requirement
when the concrete shared piece is a stub that would break your own file on
import. The *taxonomy* (`exit_code_for`) was real and safe to depend on; the
*decorator wrapper* was not.

**How to apply:** in any multi-issue epic where a context block says "use
shared seam X" and X turns out to still be `raise NotImplementedError`
(check before assuming it's real — see
[[feedback_epic_seam_convention_ownership_move]] and
[[feedback_narrowed_shared_helper_signature_drops_params]] for related
epic-seam gotchas): distinguish between the *stub you must not touch* (owned
by a sibling issue, per explicit scope instructions) and the *real,
already-implemented lower-level pieces* it would have been built from. Depend
on the real pieces directly with a local equivalent; don't block your own
GREEN-real bar on a sibling's unfinished stub, and don't silently modify a
file you were told is out of scope.
