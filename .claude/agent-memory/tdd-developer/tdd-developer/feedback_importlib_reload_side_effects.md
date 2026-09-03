---
name: importlib-reload-side-effects
description: importlib.reload changes module globals permanently; prefer patch.object over reload+patch.dict
metadata:
  type: feedback
---

`importlib.reload(mod)` re-executes the module with the current `sys.modules` state, permanently changing module-level globals (e.g. `wandb = None`). The `patch.dict(sys.modules, ...)` context manager restores `sys.modules` on exit, but does NOT undo the reload's changes to the module's own namespace.

**Consequence:** Tests that reload a module to test "absent optional" behavior will corrupt subsequent tests that expect the real module to be present.

**Fix:** Use `patch.object(module, "wandb", None)` to patch the module-level name directly. This is always reversible and doesn't cause onnx or other extension C-library re-init issues.

**How to apply:** Any time you want to test `try/except ImportError` stubs, prefer `patch.object(mod, "dep_name", None)` over `importlib.reload` + `sys.modules` patching.
