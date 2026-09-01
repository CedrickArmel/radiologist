---
name: cli-run-env-var-must-restore
description: A CLI group's run(argv) that maps a --output flag to the RADIOLOGIST_OUTPUT env var (since a @hydra.main-decorated function can't receive extra params) must save/restore the prior value in a finally block, or it leaks into every later test in the same pytest process.
metadata:
  type: feedback
---

When a `radiologist-cli` command group's `run(argv)` needs to pass
`--output`/`-o` through to a `@hydra.main`-decorated entry point (which only
ever receives `cfg` from Hydra — no room for extra params), the only channel
is `radiologist.utils.cli.output.OUTPUT_ENV_VAR` (`RADIOLOGIST_OUTPUT`),
since `emit()`/`resolve_format()` read it when no explicit `fmt` is passed.

**Why**: setting `os.environ[OUTPUT_ENV_VAR] = fmt` without restoring it
afterward leaks into every subsequent test in the same pytest process (they
all import the same interpreter's `os.environ`). This silently broke 8
unrelated `radiologist-utils/radiologist_utils_tests/test_cli_output.py`
tests (they started getting JSON output instead of the default `kv` format)
the first time the full `radiologist-etl radiologist-cli radiologist-utils`
suite ran together after a `radiologist etl --output=json` test executed
earlier in the same session.

**How to apply**: any `run(argv)` that extracts an output-format flag and
writes it to the env var for a decorated entry point to read must capture
the previous value and restore it (or delete it if it wasn't previously set)
in a `finally` block that wraps the call. Always run the full multi-package
suite together at least once (not just the new test file in isolation)
before declaring GREEN — single-file test runs won't catch this class of
leak.

See also [[prefect-broken-local-server-use-fn-bypass]] for the sibling
issue found while implementing the same `etl` command group.
