---
name: etl-retire-legacy-flow-190
description: issue #190 (final refactor, milestone #16 three-stage ETL epic) — what was actually dead vs. still-live in ops.py/prefect_pipelines.py/models.py, and how the "consolidate record-table conversion" AC was already satisfied
metadata:
  type: project
---

Implemented issue #190 on branch `feat/190-retire-legacy-etl-flow` (based on
`feat/16-etl-three-stage-framework`, NOT main), commit `91f6850`.

**Deleted `radiologist-etl/src/radiologist/etl/ops.py` entirely.** Grepping
the whole workspace (not just `radiologist-etl`) for every symbol it defined
(`compute_run_id`, `_compute_stats`, `_apply_filters`, `_assign_splits`,
`_build_shards`, `_write_jsonl`, `_df_to_records`) showed exactly one
caller: `prefect_pipelines.py`'s own dead code. `extract.py`/`assign.py`/
`build.py` (the three real stage modules from #183-#185) never imported
from `ops.py` — they already went through `manifest.py`'s
`records_reader`/`JsonlWriter`/`ManifestRecord.from_flat_dict`/
`_to_flat_dict` directly, which is genuinely the single home for
record<->table conversion. So issue #190's AC "consolidate the
record-table conversion helpers ... into a single home alongside the
manifest reader and writers" needed **zero new code** — it was already true
before this issue; the only offending duplicate was `ops.py`'s
`_df_to_records`, and deleting the whole file removed it. Don't go looking
for code to *move* on a "consolidate X" AC without first checking whether
the target home already has it and the "duplicate" is purely in the
soon-to-be-deleted file.

**In `prefect_pipelines.py`**, removed (in this order, verifying tests after
each): the `from radiologist.etl.ops import (...)` block; the five
`@task`-wrapped forwarding wrappers (`compute_stats_task`,
`apply_filters_task`, `assign_splits_task`, `write_jsonl_task`,
`build_shards_task`) — each did nothing but call its `ops.py` twin plus
an artifact call, and was only ever invoked from `etl_flow`; then
`etl_flow` itself. Kept `_haralick_list` — it looks like it belongs to the
old flow but `extract_flow` (the *live* Prefect flow, #186) also calls it.
Always check every remaining flow/task in the file for a helper before
deleting it just because it sits textually next to `etl_flow`.

**Removed now-dangling imports as a result**: `os` (only use was
`os.cpu_count()` inside `etl_flow`), `radiologist.utils.filesystem as fst`
(only used inside the five deleted `*_task` wrappers to compute artifact
keys from path stems), and `EtlResult` from the `models` import. Verified
with `grep -n "fst\."`/`"os\."` post-deletion rather than assuming.

**In `models.py`**, deleted the `EtlResult` dataclass — it was only
constructed by `etl_flow` and only imported by `prefect_pipelines.py`;
nothing else in the workspace referenced it (confirmed via a full-workspace
grep) except a test asserting it's *not* exported from the package
`__all__` (that test doesn't care whether the class itself still exists,
so it still passes either way — but since it was genuinely dead, deleted it
per the issue's "remove imports, exports and test helpers left dangling"
scope line).

**Zero test files touched** — the AC explicitly required this
("no test may be changed to accommodate this refactor — if one must change,
the scope is wrong"), and it held: #187's own memory
([[etl-cli-cutover-187]]) already documented that the `etl_flow`/`EtlResult`
tests in `test_prefect_pipelines.py` were deleted back in #187, not deferred
to #190 — so by the time #190 ran there was nothing left in the test suite
referencing any of the removed symbols. Confirmed via
`grep -rln -E "etl_flow|EtlResult|StatsProcessor|compute_run_id|_task\(|_df_to_records"`
across the whole workspace before touching anything.

**Verification**: `make test-etl` (210 passed), `make test-cli` (91 passed),
`make test-core` (128 passed, 2 deselected) all green: full suites run via
the shared `radiologist` venv's interpreter directly
(`/home/vscode/.pyenv/versions/radiologist/bin/python -m pytest <pkg> -q
--confcutdir=.`) since this worktree was disk-starved for its own venv (see
[[feedback_worktree_disk_exhaustion_reuse_shared_venv_readonly]] and
[[etl-three-stage-skeleton-180]]). `mypy radiologist-etl/src` clean except
the same pre-existing `optional.py:59` `create_markdown_artifact`
redefinition error noted in #180's memory (confirmed still pre-existing,
unrelated). `mypy radiologist-core/src` has one pre-existing unrelated
`Trainer.datamodule` attr-defined error, also not touched by this issue.

See also [[etl-cli-cutover-187]], [[etl-three-stage-skeleton-180]].
