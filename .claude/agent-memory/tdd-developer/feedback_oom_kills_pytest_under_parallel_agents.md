---
name: oom-kills-pytest-under-parallel-agents
description: Pytest runs that spawn process pools get SIGKILL (exit 137) with empty output when several agents share the box; run file-by-file unbuffered and inject a serial mapper in your own tests
metadata:
  type: feedback
---

When several agents work concurrently in worktrees on the same machine, a
pytest run whose tests spawn a `ProcessPoolExecutor` (here `local_mapper` with
`default_workers() == os.cpu_count()`) is killed by the OOM killer. Symptoms
that look like a hang or a pass but are neither:

- `EXIT=137` / `Killed` from the shell, with the redirect file **empty** —
  pytest's buffered output never flushed, so it reads as "no failures".
- A background task reported as "completed, exit code 0" because the wrapper's
  exit code is not the killed child's.
- Multi-minute "hangs" at collection when the box is thrashing on swap.

**Why:** peak RSS is the pytest process (~550 MB with the full ML stack) plus
one fully-imported worker per CPU (~450 MB each). Four workers exhausts a 8 GB
box already carrying other agents.

**How to apply:**

- Always run pytest with `python -u -m pytest` and redirect to a file, so
  partial progress survives a kill and you can tell OOM from a real failure.
- Check `free -m` before trusting a "hang"; check for `Killed`/`137` before
  trusting a silent exit 0.
- Run the suite **one test file at a time** rather than a whole directory —
  bounds peak memory and localises the kill.
- In tests you write yourself, inject the stage's `mapper` seam with a serial
  in-process helper that calls the *real* worker. That is the injection point
  the production API already exposes, not a mock of owned code:

  ```python
  def _serial_mapper(jobs):
      from radiologist.etl.shards import write_shard
      return [write_shard(job) for job in jobs]
  ```

  Pre-existing tests that use the default pool cannot be changed when the issue
  restricts which tests you may touch — only your new ones.
