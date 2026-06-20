## ✨ WandbRegistry — stage-tag (alias) management

### Context

This slice replaces the `get_aliases`, `set_alias`, and `remove_alias` stubs with real implementations over the documented W&B alias surface (`api.artifact(...).aliases` mutate-in-place then `.save()`). Independent of #2/#3 — only needs the skeleton. See `registry-spec.md`. Requires: #1. Target GREEN-real for all three alias methods.

### User story

As an **MLOps operator**, I want to read, add, and remove stage tags on a model artifact so that I can manage promotion stages without hand-writing W&B API calls.

### Acceptance criteria

- [ ] Given an artifact reference, `get_aliases` returns its current list of aliases.
- [ ] Given an artifact and a new alias, `set_alias` adds it and persists; calling it with an already-present alias leaves the alias set unchanged (idempotent).
- [ ] Given an artifact and an existing alias, `remove_alias` removes it and persists.
- [ ] Calling `remove_alias` with an absent alias is a no-op (no error, alias set unchanged).
- [ ] When wandb is not installed, all three methods raise `RuntimeError`.
- [ ] mypy clean; pytest green.

### Out of scope

- Resolve/pull (#2) and promote (#3).

### Technical notes

- Use the alias surface from the spec: `art = api.artifact(ref); art.aliases = [...]; art.save()`. Gate on the `_wandb` sentinel.
- Tests mock only the `wandb.Api()` boundary; assert observable alias state and that `.save()` was invoked when a mutation occurred.
