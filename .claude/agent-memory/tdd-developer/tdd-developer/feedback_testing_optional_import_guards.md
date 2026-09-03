---
name: testing-optional-import-guards
description: how to write a real red test for a try/except ImportError guard, not just a sentinel-patch no-op test
metadata:
  type: feedback
---

Patching `module.dep = None` via `patch.object` (the established codebase pattern, e.g. `test_export_onnx_does_not_import_or_call_wandb`) only proves the *call sites* tolerate `dep is None` — it does not prove the module actually guards the import with `try/except ImportError`. That patch works identically whether the import was guarded or bare, since both produce a module-level name that can be reassigned.

To get a genuine failing-then-passing test for the guard itself: patch `builtins.__import__` to raise `ImportError` for the target module name, then `importlib.reload()` the guarded module inside that patch. Before the guard exists, the reload propagates the `ImportError` (real red). After adding `try/except ImportError: dep = None`, the reload succeeds and `mod.dep is None` (real green). Always reload the module again in a `finally` block (outside the patch) to restore the real dependency — per [[feedback_importlib_reload_side_effects]], `importlib.reload` permanently mutates the module's globals, so a one-directional reload without restoration corrupts later tests in the same session.

**How to apply:** when an issue's acceptance criterion is "importing X succeeds when dependency Y is absent," write both: (1) the reload+blocked-import test proving the guard exists, and (2) the `patch.object(mod, "Y", None, create=True)` no-op test proving call sites tolerate it. Only the former is a true regression test for the guard.
