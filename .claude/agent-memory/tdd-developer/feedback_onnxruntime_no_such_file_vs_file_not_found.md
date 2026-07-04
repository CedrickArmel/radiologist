---
name: onnxruntime-no-such-file-vs-file-not-found
description: onnxruntime.InferenceSession raises its own NoSuchFile error for a missing model path, not Python's FileNotFoundError — check existence explicitly before constructing the session if the AC requires FileNotFoundError.
metadata:
  type: feedback
---

`onnxruntime.InferenceSession(path)` on a nonexistent path raises
`onnxruntime.capi.onnxruntime_pybind11_state.NoSuchFile`, not the stdlib `FileNotFoundError`. If an
acceptance criterion says "loading from a missing path raises FileNotFoundError," you must add an
explicit `os.path.exists(path)` check (or catch-and-reraise) before/around the `InferenceSession(...)`
call — the library's own exception type won't satisfy a `pytest.raises(FileNotFoundError)`
assertion.

**Why:** third-party inference/IO libraries typically wrap missing-file errors in their own
exception hierarchy for cross-platform/backend consistency; they do not subclass the stdlib
exception callers usually expect.

**How to apply:** whenever a spec asks for a standard Python exception type (`FileNotFoundError`,
`ValueError`, etc.) on a path/IO operation delegated to a third-party library, verify what that
library actually raises (a quick `python -c` repro) before assuming it matches — add the explicit
check/wrap in the wrapping code if it doesn't.
