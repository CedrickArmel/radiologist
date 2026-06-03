---
name: dataloader-worker-pickling
description: DataLoader workers require picklable objects; classes defined inside test functions will fail with spawn multiprocessing
metadata:
  type: feedback
---

DataLoader `num_workers > 0` spawns separate processes via `multiprocessing.spawn`. Any object passed to a worker (Dataset, collate_fn, worker_init_fn) must be picklable.

**Why:** `spawn` pickles the Dataset object to send it to the child process. A class defined inside a function body (`class Foo` inside `def test_foo`) is a closure-scoped name — pickle cannot resolve it by module path and raises `AttributeError: Can't pickle local object`.

**How to apply:** When writing tests that use DataLoader with `num_workers > 0`, define all Dataset classes at module level, not inside test functions. Alternatively, avoid `num_workers > 0` in tests — call the `worker_init_fn` directly to verify seeding behavior without spawning workers, which also avoids cross-test `sys.path` pollution in multi-package pytest runs.
