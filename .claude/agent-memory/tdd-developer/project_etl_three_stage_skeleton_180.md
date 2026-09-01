---
name: etl-three-stage-skeleton-180
description: issue #180 skeleton scope decisions and name-collision resolutions downstream issues (#181-190) in the three-stage ETL epic must know about
metadata:
  type: project
---

Implemented issue #180 (milestone #16 epic) on branch `feat/180-etl-skeleton`,
commit `a6e52f3`. Key decisions future issues in this epic (#181-190) need to
respect:

**radiologist-cli/src/radiologist/cli/groups/etl.py was NOT touched**, despite
the issue body's "Public API contract" section listing `SUBCOMMANDS`,
`_ensure_input_exists`, `extract_main`/`assign_split_main`/`build_main`, and a
rewritten `run(argv)` for that file. That file's Module layout section (the
authoritative file-placement list in #180's own body) only listed
`radiologist/etl/*` — no cli files. The new `run(argv)` contract's signature
is *identical in name* to the current production `run(argv)` that the
dispatcher (`radiologist.cli.main.run_group`) actually calls today; stubbing
it to `raise NotImplementedError` would have broken the live, tested CLI
command, violating "existing suite stays green." This work is deferred
entirely to **issue #187 (CLI cutover)**, which explicitly owns removing the
monolithic flow and swapping the command surface.

**Real, currently-used functions were NOT replaced with stubs**, even where
the epic's target contract gives them a new name/signature that will
eventually replace them:
- `radiologist.etl.split.assign_split(filename, ratios: dict[str, float])` —
  left untouched (still MD5/dict-ratios). The contract's
  `assign_split(filename, ratios: SplitRatios)` (ordered-sequence, raises
  ValueError for more cases) is real behavioral work for **issue #184**
  (assign-split stage), which must implement it via TDD, not skip straight to
  a signature swap. Added `SplitRatios` type alias + `normalize_ratios` stub
  additively in the same file.
- `radiologist.etl.shards.build_shards(manifest_path, shard_root, ratios: dict, ...)`
  — left untouched (still used by `ops.py`/`prefect_pipelines.py`/
  `test_shard_building.py`). The new pure-function `build_shards(split_manifest_path, ...)`
  lives in the new `radiologist/etl/build.py` module under the **same bare
  name but a different module path** — deliberately **not re-exported** from
  `radiologist.etl.__init__` yet, so the package-level `build_shards` still
  resolves to the old, real, shards.py implementation. **Issue #185 (build
  stage)** is where the package-level rebind to `build.build_shards` happens,
  once shards.py's old version and its callers are actually retired.
- `radiologist.etl.processors.StatsProcessor` — left in place (still used by
  `ops.py._compute_stats`, which the live `etl_flow` depends on). Added
  `process_batch` as a new additive stub in the same file; it will replace
  `StatsProcessor` once #183 (extract stage) lands.

**General pattern for this epic's remaining issues**: when the issue's own
"Public API contract" snippet reintroduces a name that already has a real,
working, tested implementation elsewhere in the current codebase, do not
blindly overwrite it with `raise NotImplementedError` just because the
contract shows that signature. Check whether the *issue's own* Module-layout
section actually places that file/change in scope; if the collision is with
production code the current test suite exercises, keep the old one working
and add the new one under a name/location that doesn't collide, documenting
the deferred rebind in a comment. This mirrors
[[feedback_stubbed_shared_decorator_blocks_sibling_slice]] and
[[feedback_skeleton_issue_repoints_old_tests]] but for plain functions/CLI
entry points, not just decorators.

**`radiologist.etl.manifest.records_reader`** — widened `storage_options` to
`dict | None = None` (small, safe, non-breaking default addition explicitly
called out as in-scope for #180, unlike the above).

**Environment**: disk-starved worktree — could not `uv sync` a dedicated
`radiologist-agent-<id>` venv (torch/nvidia wheels exceed available space,
same as [[feedback_worktree_disk_exhaustion_reuse_shared_venv_readonly]]).
Used the shared `radiologist` venv's interpreter read-only via
`/home/vscode/.pyenv/versions/radiologist/bin/python -m pytest ... --confcutdir=.`
and `.../bin/mypy radiologist-etl/src` — the root `conftest.py`'s
`sys.path.insert` (relative to `__file__`, i.e. this worktree) resolves the
worktree's own source, not the main checkout's, so no `.pth` corruption risk.
Confirmed the one mypy error in `optional.py:59` (`create_markdown_artifact`
Incompatible redefinition) is **pre-existing on the base branch** (verified
via `git stash` + rerun) — not introduced by this issue, left as-is.

**`prefect.tasks.unmapped` does not exist** in prefect 3.7.4 — the real
symbol is `prefect.task_runners.unmapped`. Fixed the sentinel import
accordingly in `radiologist.etl.optional`.
