---
name: etl-assign-split-184
description: issue #184 (milestone #16 epic) assign-split stage implementation decisions #186 (orchestration) and #190 (refactor) depend on
metadata:
  type: project
---

Implemented issue #184 on branch `feat/184-assign-split-stage` off
`feat/16-etl-three-stage-framework`, commit `62310f9`. Key facts downstream
issues (#186 orchestration, #190 refactor) need to know:

**`radiologist.etl.split.assign_split` signature changed** from
`assign_split(filename, ratios: dict[str, float])` to
`assign_split(filename, ratios: SplitRatios)` where `SplitRatios =
Sequence[tuple[str, float]]`. A plain `Mapping` now raises `ValueError`
("order is part of the split contract") instead of being silently
accepted — this is the epic's ML-correctness invariant made real.
`normalize_ratios` does the actual validation (rejects mapping, empty
sequence, repeated name, negative fraction, sum != 1.0) and is called by
`assign_split` itself, so validation happens on every call, not just at the
stage boundary.

**One pre-existing production call site broke and was fixed minimally**:
`radiologist.etl.ops._assign_splits` (the still-live monolithic flow, used
by `etl_flow`/`prefect_pipelines.py`) passes a `dict` ratios from Hydra
config into `assign_split`. Rather than change `ops.py`'s own signature
(out of #184's territory — `ops.py` wasn't in the "primary files" list but
also wasn't in the "avoid touching" list), the call site itself now does
`ordered_ratios = list(ratios.items())` before calling `assign_split`. This
is the general pattern for future issues: when a signature you own is
consumed by code outside your territory, normalize at the call site rather
than widening your own contract back toward the old one.

**`assign_splits(manifests_dir, destination, ratios=None, run_label=None,
storage_options=None) -> AssignSplitResult`** implemented in `assign.py`:
- Lists `manifests_dir` twice per run (once via
  `compute_assign_run_id`→`directory_digest`, once directly to enumerate
  `.jsonl` files) — a deliberate, documented tradeoff per the issue's own
  technical notes, not an oversight to "fix" later.
- Dedup key is `record.path`; manifests are read in **sorted name order**
  and first occurrence wins — this is what makes duplicate resolution
  deterministic regardless of `fs.ls` ordering.
- Filename-collision warning (same filename, different path — a shard-key
  hazard) is **separate** from the duplicate-path warning (one combined
  `logger.warning` with the dropped count, not one warning per duplicate).
- `manifest_id` on every written record is overwritten to the new
  `assign_splits` run id (via `dataclasses.replace`), not left as the
  extract-stage's run id — the split manifest is its own run's artifact.
- Missing-folder and empty-folder (no `.jsonl` files) both raise the same
  `FileNotFoundError(f"No extract manifest found in {manifests_dir!r}")` —
  unified handling, not two separate code paths.
- Idempotency/content-addressing falls out "for free" from
  `compute_assign_run_id` + deterministic dedup/assignment — no explicit
  byte-identical-write special-casing was needed; the AC test simply reran
  `assign_splits` twice and diffed file bytes.
- Default ratios `[("train", 0.70), ("val", 0.15), ("test", 0.15)]` were
  verified against `conf/etl.yaml`'s existing `split_ratios:` mapping
  (train/val/test in that exact key order) and `conf/build.yaml`'s
  already-shipped ordered list (from skeleton #180) before being hardcoded
  — both matched, confirming no re-partitioning risk.

**Test file**: `radiologist-etl/radiologist_etl_tests/test_assign_split_stage.py`
(18 tests, all through the public `radiologist.etl.assign_splits` API).
`test_split_assignment.py` was **rewritten in place** (not left broken) to
use ordered-pair ratios instead of dict ratios, per
[[feedback_skeleton_issue_repoints_old_tests]] — the old dict-based tests
would have failed against the new signature.

**mypy gotcha**: the pre-commit mypy hook checks test files too (unlike a
manual `mypy src/` invocation scoped to source only) — a `dict(...)`
literal without an explicit `dict[str, object]` annotation, later
`.update()`-d with a `**overrides: object` kwargs dict, triggered a
`MutableMapping` argument-type error only inside the pre-commit hook run,
not in the manual `mypy radiologist-etl/src` check. Always run the
pre-commit hook itself (or `mypy` over the full package including tests)
before trusting a source-only mypy pass as "clean."

**Environment**: same disk-starved worktree constraint as
[[project_etl_three_stage_skeleton_180]] — used the shared `radiologist`
venv's interpreter read-only via
`/home/vscode/.pyenv/versions/radiologist/bin/{python,mypy,black,isort,flake8}`
with `--confcutdir=.` for pytest, and `PATH=".../bin:$PATH" git commit` for
the pre-commit hooks (GPG signing worked fine).
