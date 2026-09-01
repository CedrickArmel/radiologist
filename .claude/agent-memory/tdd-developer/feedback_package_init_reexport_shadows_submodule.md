---
name: package-init-reexport-shadows-submodule
description: a package __init__.py that does `from pkg.mod import mod_func` (same name as the submodule) makes `import pkg.mod as x` fetch the re-exported object, not the submodule, once the package has been imported
metadata:
  type: feedback
---

When a package's `__init__.py` re-exports a name that collides with one of its
own submodule names (e.g. `radiologist/cli/__init__.py` does
`from radiologist.cli.main import main`), Python's attribute-based import
resolution for `import pkg.submodule as x` fetches whatever is currently bound
to `pkg.submodule` in the package namespace — which the `__init__.py` import
just overwrote with the re-exported function/class. The submodule *object*
still exists in `sys.modules["pkg.submodule"]`, but `import pkg.submodule as x`
silently returns the shadowing function instead once the package `__init__`
has run (which it always has, by the time any test file executes).

**Why:** hit this writing dispatcher tests for issue #176 — `import
radiologist.cli.main as main_module` returned the `main` *function*, not the
module, because `radiologist/cli/__init__.py` does
`from radiologist.cli.main import GROUPS, main, run_group`. `monkeypatch.setattr(main_module, "run_group", ...)` then failed with
`AttributeError: <function main> has no attribute 'run_group'`.

**How to apply:** whenever a package's public `__all__`/`__init__.py`
re-exports a symbol whose name matches one of its submodules, fetch the
submodule in tests via `importlib.import_module("pkg.submodule")` (reads
`sys.modules` directly, bypassing the shadowed package attribute) rather than
`import pkg.submodule as x`. This is a general Python import-system fact, not
specific to this project — applies to any package with a submodule-shadowing
re-export.
