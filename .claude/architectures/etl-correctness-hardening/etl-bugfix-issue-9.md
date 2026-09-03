## ✨ ETL CLI result payloads omit the fields operators gate on

**Requires:** #1 · **Blocks:** —

### Context

Each of the three ETL subcommands prints a machine-readable result record when
it succeeds. Those records are the only programmatic view a scheduler, a CI job
or an operator has of what a run actually did — and each of the three drops a
field the stage already computed and returned.

- **extract** prints `run_id`, `manifest_path`, `total`, `succeeded`, `failed`
  and `excluded`, but not the **failure rate**. The failure rate is the value
  the stage itself gates on; omitting it forces every consumer to recompute it
  from two other fields and handle the zero-total case themselves.
- **assign-split** prints `run_id`, `split_manifest_path`,
  `source_manifest_count`, `record_count` and `duplicate_count`, but not the
  **per-split counts**. A caller cannot tell whether a split came out empty —
  the single most common reason a downstream training run fails immediately.
- **build** prints `run_id`, `output_dir`, `manifest_path`, `report_path` and
  `shard_count`, but neither the **record count** nor the **failure count**. So a
  build that wrote shards containing zero usable records is indistinguishable,
  from the outside, from a healthy one.

This is not a defect in the stage functions — every one of these values is
already on the result object each stage returns. It is a gap in what the command
layer forwards.

### User story

As an **ML engineer running the ETL pipeline from a scheduler**, I want each
stage's printed result to carry the counts and rates the stage itself computed,
so that **I can gate the next stage on the previous one's output without
re-deriving it or re-reading the manifest**.

### Behaviour to implement

Extend the three `emit(...)` payloads in
`radiologist-cli/src/radiologist/cli/groups/etl.py`, adding only fields the
corresponding stage result already carries. Preserve every existing field and
its name — these are consumed contracts.

**extract** (`emit` at `:105-114`) — add `failure_rate`, the stage's already
computed `failed / total` (`0.0` when the total is zero), alongside the existing
`run_id`, `manifest_path`, `total`, `succeeded`, `failed`, `excluded`.

**assign-split** (`emit` at `:138-146`) — add `counts_by_split`, a mapping of
split name to record count, alongside the existing `run_id`,
`split_manifest_path`, `source_manifest_count`, `record_count`,
`duplicate_count`.

**build** (`emit` at `:170-178`) — add `record_count` and `failed`, alongside
the existing `run_id`, `output_dir`, `manifest_path`, `report_path`,
`shard_count`.

The `emit` helper already handles nested mappings in all three supported
formats: for `kv` it flattens them into dotted leaf keys, and for `json` / `yaml`
it serialises them structurally. `counts_by_split` therefore needs no special
handling — **do not flatten it by hand at the call site.**

### Acceptance criteria

- [ ] A successful extract run's printed result carries a failure rate equal to
      the failed count divided by the total count, and carries every field it
      carries today.
- [ ] A successful extract run over a listing whose images are all readable
      prints a failure rate of `0.0`.
- [ ] A successful assign-split run's printed result carries a per-split count
      for every split named in the configured split ratios, including splits
      that received zero records, and carries every field it carries today.
- [ ] A successful build run's printed result carries a record count and a
      failure count, and carries every field it carries today.
- [ ] In the key-value output format, a successful assign-split run's per-split
      counts appear as one line per split, keyed by split name.
- [ ] In the JSON output format, a successful assign-split run's per-split counts
      appear as a nested object keyed by split name, not as flattened keys.
- [ ] A run that fails still exits with its existing exit code and prints no
      result record — unchanged from today. In particular, a missing input still
      exits `2`, and any other error still exits `1`.
- [ ] mypy clean; pytest green

### Out of scope

- Renaming or removing any existing field.
- Adding fields to the stage result objects themselves. Every value this issue
  forwards already exists on them: `ExtractResult` carries `failure_rate`
  (`models.py:51`), `AssignSplitResult` carries `counts_by_split`
  (`models.py:73`), and `BuildResult` carries `record_count` (`models.py:94`) —
  plus `failed`, added by #1.
- Any change to exit-code mapping, input-existence checking, or the
  output-format flag.
- Documentation outside `radiologist-cli`.
- Any change to the `emit` helper itself.

### Technical notes

- `radiologist-cli/src/radiologist/cli/groups/etl.py` — this issue is the sole
  owner of this file for the epic. The three changes are three additions to
  three existing result mappings; nothing else in the module changes.
- The build result's `failed` field was added, defaulted to `0`, by #1. Its
  value only becomes non-zero once #2 lands; **this issue does not depend on
  #2**, because the field exists and reads `0` in the meantime, and the
  acceptance criterion above asserts the field's presence rather than a
  particular non-zero value.
- This module is a `radiologist-cli` module and does **not** carry
  `from __future__ import annotations` — it uses `Optional[...]` / `Dict[...]`
  from `typing`. Match the file's existing style, not the ETL package's.
- The output helper flattens an **empty** mapping to a single `key=` line rather
  than one line per entry. `counts_by_split` is always non-empty (the
  assign-split stage seeds one entry per configured ratio before counting), so
  the "one line per split" criterion is well-defined — but do not write a test
  that depends on the empty-mapping case.
- The CLI tests for this group live in
  `radiologist-cli/radiologist_cli_tests/test_etl_command.py` and drive the
  group's `run(argv)` entry point, capturing stdout. Extend the existing
  `test_*_subcommand_prints_record_with_run_id_and_counts` tests and their
  neighbours there. Output-format behaviour is covered alongside them (see the
  `test_output_flag_honoured_*` tests for how the format flag is exercised).
- Do not mock the ETL package — it is owned code. Run the real stages over the
  shared tiny-PNG fixture corpus, which is how the existing tests in that file
  already work.
- Docs: this issue changes the shape of documented command output. Update the
  `etl` group module's own docstring if it enumerates the emitted fields, and any
  `radiologist-cli/README.md` table that does. Do **not** touch
  `radiologist-etl/README.md` — four concurrently-landing issues own sections of
  it.
