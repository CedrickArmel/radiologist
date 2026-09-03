---
name: hydra-full-error-for-exit-codes
description: Hydra's run_and_report() swallows every job exception into a bare sys.exit(1) unless HYDRA_FULL_ERROR=1 is set; needed whenever a @hydra.main entry point's CLI wrapper must map exception types to distinct process exit codes.
metadata:
  type: feedback
---

Without `HYDRA_FULL_ERROR=1` in the environment, Hydra's internal
`run_and_report()` catches *any* exception raised inside the decorated
task function, pretty-prints a stripped traceback, and always calls
`sys.exit(1)` — regardless of the exception type. This defeats any outer
wrapper that tries to inspect the real exception (e.g. `FileNotFoundError`
-> exit 2 vs. other -> exit 1) since the original exception never
propagates out of the `@hydra.main`-decorated call.

**Why:** discovered while implementing `radiologist-cli`'s `core train`
command (issue #175): a `run(argv)` wrapper set `sys.argv` and called the
Hydra-decorated `train_main()` in a `try/except`, expecting to catch
`FileNotFoundError` for a missing checkpoint and map it to exit code 2.
It always got exit 1 with Hydra's own pretty-printed message until
`os.environ["HYDRA_FULL_ERROR"] = "1"` was set immediately before the
call (and restored/cleared in a `finally`).

**How to apply:** any `run(argv)`-style wrapper around a `@hydra.main`
entry point that needs per-exception-type exit codes (or just the raw
exception message instead of Hydra's traceback-stripped one) must set
`HYDRA_FULL_ERROR=1` around the call to `train_main()`/decorated
function. `--help`/`SystemExit` paths are unaffected (argparse calls
`sys.exit` directly, not through `run_and_report`).

See also [[hydra-cfg-passthrough-for-tests]], [[shared/MEMORY.md]].
