---
name: spawn-pool-needs-pythonpath-single-file
description: running a single test file directly (bypassing the package-level pytest invocation) breaks its ProcessPoolExecutor(spawn) tests with ModuleNotFoundError unless PYTHONPATH includes the test package dir
metadata:
  type: feedback
---

A test file whose tests exercise a `spawn`-context `ProcessPoolExecutor` (e.g.
`radiologist.etl.execution.local_mapper`) needs its own test package
(`radiologist_etl_tests`) importable by the **spawned child** process, not
just by the parent pytest process. Running the *whole* test directory via
`uv run --active pytest <pkg>/radiologist_etl_tests` works because uv's run
context puts the package on `sys.path` for the whole tree. Running a **single
file directly** (e.g. `python -u -m pytest .../test_execution.py`, done to
localize an OOM per [[feedback_oom_kills_pytest_under_parallel_agents]]) does
not carry that onto `PYTHONPATH`, so the spawned worker fails with
`ModuleNotFoundError: No module named 'radiologist_etl_tests'` and the whole
pool becomes a `BrokenProcessPool` — a real-looking `1 failed, N passed`, not
a flake.

**How to apply:** when isolating a single file to dodge OOM/contention, set
`PYTHONPATH=<package-dir>` (e.g. `PYTHONPATH=radiologist-etl`) on the same
invocation. Confirms the failure is an artifact of the isolation technique,
not a regression, before spending time investigating the "failure".
