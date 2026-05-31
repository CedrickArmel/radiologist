---
name: feedback-write-files-via-python
description: When writing source files that contain != or other shell-special chars, use python3 -c or a Python script instead of heredoc cat
metadata:
  type: feedback
---

Use `python3 -c` (with a PYEOF delimiter) or the Write tool to write source files that contain `!=`, `!`, or other shell metacharacters.

**Why:** bash heredocs process `!` as a history-expansion character even inside single-quoted delimiters in some shell configs, turning `!=` into `\!=` which is a Python syntax error.

**How to apply:** Whenever a file to be written contains `!=`, `!`, or `\` sequences, write it via the Python `open()` approach: `python3 - << 'PYEOF' ... PYEOF` with the content as a Python string literal. If the Edit tool is available for the path, prefer it for surgical fixes.
