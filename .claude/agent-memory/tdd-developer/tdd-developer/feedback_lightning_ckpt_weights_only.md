---
name: lightning-ckpt-weights-only
description: LModule.load_from_checkpoint requires weights_only=False in PyTorch 2.6+ because checkpoints contain functools.partial
metadata:
  type: feedback
---

PyTorch 2.6 changed `torch.load` default to `weights_only=True`. Lightning checkpoints embed Python objects (e.g. `functools.partial` factories from `save_hyperparameters`) that are not safe globals, so `load_from_checkpoint` raises `UnpicklingError` unless `weights_only=False` is passed explicitly.

**Why:** `LModule` stores optimizer/metric/scheduler as `functools.partial` in hparams. These can't be deserialized with `weights_only=True`.

**How to apply:** Always call `LModule.load_from_checkpoint(path, map_location="cpu", weights_only=False)` in production code. In tests, mock `LModule.load_from_checkpoint` with `patch.object` to return a pre-built instance — this avoids the serialization issue entirely and is the cleaner test approach.
