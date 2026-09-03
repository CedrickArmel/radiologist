---
name: feedback-hydra-main-test-via-subprocess
description: test a @hydra.main-decorated entry point's --help/exit-0 behavior via subprocess, not in-process sys.argv monkeypatching
metadata:
  type: feedback
---

Calling a `@hydra.main(config_path="conf", ...)`-decorated function directly inside a running
pytest process (after `monkeypatch.setattr(sys, "argv", [...])`) raised
`MissingConfigException: Primary config module '<pkg>.conf' not found` even though the config
directory exists on disk and the same invocation works when run as a real process
(`python -m <pkg>.module --help` from the package root, exit 0). Hydra's config-module
resolution depends on `pkgutil`/import-machinery state that differs between "imported inside an
already-running interpreter with pytest's sys.path shims" and "launched fresh via `-m`".

**Why:** this matches the project's own testing philosophy — a console script's CLI behavior is
a true process boundary, not something to validate via in-process monkeypatching. Treating it as
an in-process call also produces a flaky, environment-sensitive failure that has nothing to do
with the actual code under test.

**How to apply:** for any acceptance criterion phrased as "the CLI/console script exits 0 on
`--help`" (or similar shell-level behavior of a Hydra/Typer entry point), assert it with
`subprocess.run([sys.executable, "-m", "<package>.<module>", "--help"], cwd=<package_root>,
capture_output=True, text=True)` and check `returncode == 0` (plus a stdout substring like
"Powered by Hydra"). Don't import the decorated function and monkeypatch `sys.argv` — it doesn't
reproduce a real console-script invocation and can fail on config resolution for reasons
unrelated to the change being tested. See [[shared/feedback_tdd]] for the general "mock only
true process boundaries" rule this is an instance of.
