---
name: parallel-output-namespacing
description: In Mode A parallel exploration, namespace output files by strategy to avoid clobbering sibling agents' drafts in the shared .claude/architectures dir
metadata:
  type: feedback
---

When spawned in Mode A (parallel exploration) alongside sibling code-architect agents, all siblings write into the same `.claude/architectures/` directory. Generic filenames like `<feature>-spec.md` / `<feature>-issue-N.md` collide — a sibling's file may already occupy the name, and the Write tool then rejects the write as "file has not been read yet".

**Why:** the skill runs 2–3 strategy agents concurrently against one shared dir; there is no per-agent scratch space by default.

**How to apply:** prefix every draft file with the assigned strategy, e.g. `predictor-minimal-spec.md`, `predictor-minimal-issue-1.md` for `minimal-impact`. Report the exact paths back in the summary so the synthesis step (Mode B) can find them. Do not delete or overwrite a sibling's differently-prefixed files.
