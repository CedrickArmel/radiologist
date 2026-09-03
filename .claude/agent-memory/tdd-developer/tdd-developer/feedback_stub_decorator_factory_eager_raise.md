---
name: stub-decorator-factory-eager-raise
description: a skeleton-stubbed decorator that "raise NotImplementedError" in its own body breaks every module that applies it at import time
metadata:
  type: feedback
---

When an interface contract specifies a decorator stub like
`def exit_on_error(func: F) -> F: raise NotImplementedError`, applying that
decorator with `@exit_on_error` to command functions in the same skeleton
issue crashes at **import time** — the decorator body executes immediately
during module load (decoration happens once per definition, not per call),
so the whole module (and any Typer `app` it defines) never finishes
importing. This silently breaks an explicit AC like "the CLI's `--help`
must render the full command grammar" even though every individual command
function still has its own `raise NotImplementedError` body.

**Why:** discovered in the radiologist-cli skeleton (issue #170): decorating
stub `@app.command()` functions with `@exit_on_error` (itself a
`NotImplementedError` stub) made `from radiologist.cli.groups import
registry` raise immediately, so `typer.testing.CliRunner().invoke(app,
["--help"])` never got a working `app` to invoke against.

**How to apply:** when a skeleton issue's contract stubs both (a) a
decorator and (b) callers that would apply that decorator, do not wire the
decorator onto the stub call sites yet — leave the plain function stubs
undecorated. Only decorate once the decorator itself does real wrapping
(defers the raise to call time via a returned closure), which is a later
issue's job. Verify by actually invoking `CliRunner().invoke(app,
["--help"])` (exit code 0) for every Typer-backed group, not just a bare
`import` — a bare import can succeed even when decoration born from a
different bug would still break `--help`.
