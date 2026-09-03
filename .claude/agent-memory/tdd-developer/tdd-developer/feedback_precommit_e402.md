---
name: feedback_precommit_e402
description: Test files with sys.path.insert before imports need noqa: E402 on the import line to pass flake8
metadata:
  type: feedback
---

In `radiologist-core/tests/`, `conftest.py` uses `sys.path.insert(0, src_path)` before imports. Any test file that follows this pattern and then imports from `radiologist.*` will trigger flake8 E402 (module level import not at top of file).

**Why:** flake8 enforces imports at the top; sys.path manipulation must precede the import but is itself not an import statement.

**How to apply:** add `# noqa: E402` to the import line immediately following `sys.path.insert(...)` in any new test file, e.g.: `from radiologist.core.data.shards import _discover_shards  # noqa: E402`
