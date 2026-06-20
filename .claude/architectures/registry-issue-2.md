## ✨ WandbRegistry — resolve & pull checkpoints

### Context

This slice replaces the skeleton stubs for `WandbRegistry.resolve_checkpoint` and `WandbRegistry.pull`, porting the resolution logic that currently lives in `radiologist-core/.../registry/pull.py` (`_resolve_model_artifact` + `pull_checkpoint`) verbatim into the new package. The three resolution branches (run_id, tags+metric, raw path) and the `.ckpt` glob are exercised through the two public methods — the internal resolver helper is not a separate issue. See `registry-spec.md`. Requires: #1. Target GREEN-real: no `NotImplementedError` reachable through `resolve_checkpoint` or `pull`.

### User story

As an **ML engineer**, I want to resolve and download a training checkpoint from W&B through one registry object so that pulling no longer lives in core and obeys the same selection rules as before.

### Acceptance criteria

- [ ] Given a `run_id` and `path`, `resolve_checkpoint` returns the artifact named `{path}/model-{run_id}:best` (or `:{version}` when `version` is given).
- [ ] Given `tags` (and optional `groups`/`metric`), `resolve_checkpoint` selects the best run by the metric (default `best_val_score`) and returns its `model-{run.id}` artifact name.
- [ ] Given neither `run_id` nor `tags`, `resolve_checkpoint` returns the artifact for the raw `path` unchanged.
- [ ] When wandb is not installed, `resolve_checkpoint` and `pull` raise `RuntimeError` naming the missing dependency.
- [ ] Given a resolvable artifact containing a `.ckpt`, `pull` downloads into `local_dir` and returns the local `.ckpt` path.
- [ ] When the downloaded artifact contains no `.ckpt`, `pull` raises `FileNotFoundError`.
- [ ] `pull` forwards resolution kwargs (run_id/tags/etc.) so the same selection rules apply.
- [ ] mypy clean; pytest green.

### Out of scope

- Promotion and alias management (#3, #4).
- Any change to resolution semantics — behavior must match the old `pull_checkpoint` exactly.

### Technical notes

- Port `_resolve_model_artifact` (pull.py:35-72) and the `.ckpt` glob (pull.py:96-102) into `wandb.py`; gate every public method on the `_wandb` sentinel from `optional.py`.
- Tests mock only the `wandb.Api()` boundary (the one true process boundary); the resolver branches are reached through the public methods, not mocked.
