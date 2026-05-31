---
name: feedback-output-dir-creation
description: Code that writes to a user-specified output path must create parent directories itself — never assume they exist
metadata:
  type: feedback
---

Any function that writes files to a caller-supplied output directory must call `Path(output_dir).mkdir(parents=True, exist_ok=True)` before the first write.

**Why:** The caller controls the path and may pass a directory that does not yet exist (e.g. a fresh `tmp_path` subdirectory in tests, or a first run in CI). Backends like fsspec's local filesystem raise `FileNotFoundError` if parent directories are absent — they do not auto-create them.

**How to apply:** Wherever a new output path parameter is introduced, add the `mkdir` call at the top of the write block. Do not rely on the caller or the filesystem backend to create parents.
