---
name: feedback-gh-cli
description: Use gh CLI directly for GitHub operations — the pr-open skill hits a sandbox permission error on command substitution
metadata:
  type: feedback
---

Use `gh pr create` directly instead of the pr-open skill for this project.

**Why:** The pr-open skill fails with "Shell command permission check failed: Contains command_substitution" in this sandbox.

**How to apply:** When opening PRs, call `gh pr create --title "..." --body "$(cat <<'EOF'...EOF)"` directly. Check `gh pr list` first to avoid duplicates.
