---
name: feedback-wandb-sandbox-env-vars
description: wandb SDK writes to ~/Library/Application Support/wandb (or ~/.cache/wandb) by default, which the sandbox denies — set WANDB_DIR/WANDB_DATA_DIR/WANDB_CACHE_DIR/WANDB_CONFIG_DIR to a writable tmp dir before running radiologist-registry or radiologist-core tests that exercise real `wandb.Artifact` objects
metadata:
  type: feedback
---

Tests that instantiate a real `wandb.Artifact(...)` (not a mock) — e.g.
`_WandbUploader.log_model_artifacts` tests that call `.add_file()` — make the
wandb SDK stage the file via `tempfile.NamedTemporaryFile` under
`~/Library/Application Support/wandb/artifacts/staging` (or the platform
equivalent). The sandbox denies writes there, so these tests fail with
`PermissionError: Operation not permitted`, even though nothing about the
test or the code under test is wrong — the code being mocked is just the
`_wandb` sentinel's `Api`/network calls, not `wandb.Artifact` itself, which is
real local-only object construction.

**Why:** discovered while implementing issue #111 — pre-existing tests in
`test_artifact_promotion.py::TestUploaderLogModelArtifacts` (from #109) failed
with this `PermissionError` on a fresh run in this worktree, unrelated to the
new promotion code being added.

**How to apply:** before running `radiologist-registry` (or any package)
tests that touch real `wandb.Artifact`/`wandb.init` objects, export these to
a sandbox-writable tmp dir:

```bash
export WANDB_DIR=$TMPDIR/wandb WANDB_DATA_DIR=$TMPDIR/wandb_data \
       WANDB_CACHE_DIR=$TMPDIR/wandb_cache WANDB_CONFIG_DIR=$TMPDIR/wandb_config
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"
```

Then run pytest as usual (see [[feedback_pytest_worktree_isolation]] for the
`--confcutdir=.` requirement in worktrees). Do this proactively at the start
of any registry/core test session rather than waiting for the `PermissionError`
to appear.
