---
name: typer-cli-wandb-process-boundary
description: Testing a Typer CLI command that itself calls an SDK (e.g. wandb.init) directly, not just through an owned facade class
metadata:
  type: feedback
---

When a CLI command file imports an SDK sentinel directly (e.g. `from
radiologist.registry.optional import _wandb` inside `cli.py`, to call
`_wandb.init(...)`) in addition to going through an owned facade
(`WandbRegistry`), the CLI module's own `_wandb` reference must be patched
separately from the facade's submodule `_wandb` (e.g.
`radiologist.registry.uploader._wandb`). They are distinct module-level
bindings even though they originate from the same `optional.py` import —
patching one does not affect the other.

**Why:** `_guard_wandb()` in `optional.py` checks `optional._wandb`, the
module-global in `optional.py`, not the per-submodule imported reference. So
as long as the real SDK package is installed in the test env, `_guard_wandb()`
passes regardless of what submodule-local `_wandb` names are patched — this
is why existing tests could patch e.g. `uploader._wandb` freely without ever
touching `optional._wandb` and still pass the guard.

**How to apply:** when a CLI file both delegates to an owned facade class and
also makes a direct SDK call (init/login/etc.) itself, patch
`<cli_module>._wandb` for the direct call and patch each collaborating
submodule's `_wandb` (`resolver`, `uploader`, `collection`, `alias_manager`,
...) independently for calls that happen inside the facade. Don't assume
patching one covers the other.
