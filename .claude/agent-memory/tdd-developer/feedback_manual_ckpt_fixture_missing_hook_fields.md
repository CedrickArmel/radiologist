---
name: manual-ckpt-fixture-missing-hook-fields
description: shared test checkpoint fixtures built by hand (not via a real trainer save) can lack fields that Lightning hooks add — check before relying on them
metadata:
  type: feedback
---

The `ckpt_path` fixture in `radiologist-core/tests/conftest.py` builds its checkpoint
dict by hand (`{"epoch":..., "state_dict":..., "hyper_parameters":...}`) rather than
running a real `trainer.fit()` + save. Any field a `LightningModule` hook adds at save
time (e.g. `LModule.on_save_checkpoint` setting `checkpoint["precision"]`) will be
absent from this hand-built dict unless added explicitly.

**Why:** discovered while implementing issue #110 (`restore_precision`) — the fixture
didn't carry a `"precision"` key even though the module's `on_save_checkpoint` always
sets one, because the fixture never exercises that hook.

**How to apply:** before writing a test that reads a checkpoint field added by a
Lightning hook, check the shared fixture's checkpoint dict for that field. If missing,
extend the fixture directly (add the key with a representative value, e.g.
`"precision": "32-true"`) rather than duplicating checkpoint construction in the new
test file — keeps the "real, shared, Lightning-loadable checkpoint" fixture invariant
intact for all consumers. See [[shared/MEMORY.md]] for the broader "never mock owned
code" testing rule this fixture pattern serves.
