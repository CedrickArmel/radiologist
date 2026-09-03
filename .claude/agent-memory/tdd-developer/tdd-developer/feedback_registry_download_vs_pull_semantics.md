---
name: registry-download-vs-pull-semantics
description: WandbRegistry.download() returns a .ckpt path (training resume); .pull() returns a .onnx path (inference load) — don't conflate them even if a spec says "download"
metadata:
  type: project
---

`radiologist.registry.resolver._WandbResolver.download(ref, local_dir)` globs for `*.ckpt`
files only (used by `radiologist-core/resume.py` to resume training from a checkpoint
artifact). `.pull(artifact_path, local_dir)` globs for `*.onnx` files only (used by
inference's `from_registry`/`from_selector` to load an ONNX predictor). They are not
interchangeable despite both taking a similar shape.

**Why:** issue #125's spec described `BasePredictor.from_selector` as
`resolve_selector(selector, registry) -> ArtifactRef` then
`registry.download(ref, local_dir)` then `cls.from_path(det_path=...)`. Taken literally
this breaks — `download()` will raise `FileNotFoundError` on any artifact directory that
only contains an `.onnx` file. The correct pairing (confirmed against
`radiologist-registry/tests/test_artifact_resolution.py`'s `TestDownload`/`TestPull`
classes and `from_registry`'s existing working implementation) is:
`ref = resolve_selector(selector, reg)` then
`reg.pull(artifact_path=ref.qualified_name, local_dir=local_dir)` — `pull()` accepts a
qualified artifact path string, and `ArtifactRef.qualified_name` is exactly that.

**How to apply:** any future work resolving a `RegistrySelector` (or `ArtifactRef`) into
an ONNX file for `BasePredictor.from_path` must call `.pull(artifact_path=ref.qualified_name, ...)`,
never `.download(ref, ...)`. Reserve `.download()` for checkpoint-resume flows in
radiologist-core. If a future spec says "download" in this context, verify against the
registry's own tests before implementing literally — the two methods search for different
file extensions and are not fungible.

See also [[shared/feedback_tdd.md]].
