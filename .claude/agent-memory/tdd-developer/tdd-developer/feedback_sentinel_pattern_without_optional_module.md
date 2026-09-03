---
name: sentinel-pattern-without-optional-module
description: guard-import sentinel for a soft dependency can live directly in the module that uses it, not always in a dedicated optional.py
metadata:
  type: feedback
---

The `_wandb`/`_typer` sentinel pattern (registry/inference `optional.py`) is
about the *try/except guard + RuntimeError-on-use* shape, not about the file
it lives in. When a package has no existing `optional.py` convention and
only one guarded import is needed, put the `try: import X as _x / except
ImportError: _x = None` sentinel directly in the module that needs it
(e.g. `radiologist-utils/src/radiologist/utils/cli/output.py`'s `_yaml`
sentinel) rather than inventing a new `optional.py` file for a single guard.

**Why:** issue #171's technical notes made this call explicitly — "a single
guarded import doesn't justify a new file." Tests still patch the sentinel
directly (`monkeypatch.setattr(output_module, "_yaml", None)`), same as the
`_wandb` pattern.

**How to apply:** when a soft dependency backs a `RuntimeError`-naming-an-extra
guard, check whether the package already has an `optional.py` convention
(registry, inference do). If not, and only one guard is needed, put the
sentinel inline in the consuming module. Also: when a dependency becomes
truly optional this way, move it out of the package's hard `dependencies`
list into a matching `[project.optional-dependencies]` extra (e.g. PyYAML
moved to `radiologist-utils[cli]`), or the "missing" branch is untestable
except by monkeypatching the sentinel — which is fine and expected, but the
pyproject extra should still exist so the RuntimeError's install hint is
truthful.
