---
name: feedback-tdd
description: Always write failing tests before implementing any feature or change — strict TDD, no exceptions
metadata:
  type: feedback
---

Write the failing test first. Watch it fail. Only then write the implementation that makes it pass.

**Why:** The user enforces strict TDD. Writing tests after implementation produces tests that fit existing code rather than tests that specify intended behaviour.

**How to apply:** For every new function, class, or behaviour change: write (or update) the test file first, run the suite and confirm the new tests fail with the expected error, then implement the production code. Applies even to small additions like a new helper inside an existing module.
