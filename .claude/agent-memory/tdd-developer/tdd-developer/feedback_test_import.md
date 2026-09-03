---
name: feedback_test_import
description: Use lazy imports inside test methods to avoid collection-time ImportError for not-yet-implemented functions
metadata:
  type: feedback
---

When writing RED-phase tests for functions that don't exist yet, importing them at module level causes `ImportError` during test collection — pytest reports an ERROR rather than a FAILED test. This is the "wrong failure loop" in TDD.

**Why:** collection-time errors prevent pytest from even running the tests, so you can never watch them fail for the right reason.

**How to apply:** put the `from module import _new_function` inside the test method body for any function that doesn't exist yet. Once implementation exists and all tests pass, the lazy import can remain (it works fine) or be promoted to module level if preferred.
