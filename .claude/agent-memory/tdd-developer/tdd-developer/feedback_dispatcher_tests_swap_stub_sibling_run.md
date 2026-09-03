---
name: dispatcher-tests-swap-stub-sibling-run
description: when an epic's dispatcher issue is built before sibling command-group issues implement their run(argv) bodies, drive dispatch mechanics by swapping the sibling module's public run function rather than waiting for it to be real
metadata:
  type: feedback
---

In a phase-parallel epic where a dispatcher issue (radiologist#176) routes to
sibling command-group modules still stubbed with `raise NotImplementedError`
(owned by other issues, e.g. #172-#175), you cannot reach GREEN-real by
calling through to their real bodies — they don't exist yet, by design,
outside your scope.

**Why:** the epic context block explicitly sanctioned this: "you can drive
main()/run_group()/extract_output_flag()/split_group() behavior directly" —
meaning it is correct, not a compliance shortcut, to
`monkeypatch.setattr(sibling_group_module, "run", fake)` when testing the
dispatcher's own forwarding/mapping/exit-code-propagation contract, and
separately to `monkeypatch.setattr(main_module, "run_group", fake)` when
testing `main()`'s own env-var-lifecycle/exit-code contract in isolation from
dispatch mechanics (since `run_group` gets its own dedicated tests). This is
different from "mocking what you own" in the CLAUDE.md sense — the sibling
group bodies are out of this issue's scope entirely, not collaborators your
own GREEN-real bar depends on.

**How to apply:** in any epic with a "dispatcher built first, consumers land
in parallel" shape, check the issue's own context block for explicit
authorization language like this before assuming you must implement/mock
around not-yet-built collaborators. If authorized, swap the sibling's public
function (not a private/internal one) so the test still asserts through the
real public API surface once siblings land. See also
[[feedback_package_init_reexport_shadows_submodule]] for the import gotcha hit
while writing these tests.
