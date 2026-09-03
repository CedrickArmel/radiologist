# 🚀 Epic — `radiologist-etl` correctness hardening (ten verified defects)

## Problem Statement

Ten verified defects in `radiologist-etl` — and in the `radiologist-cli` surface
that fronts it — let the pipeline silently emit a manifest that overstates the
usable training set, refuse to start on a plain no-extras install, silently
discard explicitly configured values, record shard paths that do not resolve,
and leak unbounded scratch objects into remote object storage.

## Goal

A plain (no-extras) install runs all three ETL stages end to end on the shipped
default configuration; every stage either produces a manifest whose invariants
hold or fails loudly; run ids are reproducible and location-independent; every
shipped config key is honoured; no dispatch leaves scratch behind.

## Scope

**In scope**

- All ten verified defects in `radiologist-etl`.
- `radiologist-cli`: relaxing the `etl` extra gate to module-importability only
  (#5), and extending the three `emit(...)` payloads (#9). `radiologist-cli`
  tests are in scope for both.
- One new config key: `max_failure_rate` in
  `radiologist-etl/src/radiologist/etl/conf/build.yaml` (#2).
- Doc lines that a specific fix invalidates — each issue amends only its own.

**Out of scope**

- **The broader ETL doc rot.** Two documents describe a pipeline that no longer
  exists and neither is touched by this epic:
  - `docs/reference/config-etl.md` (84 lines) documents
    `radiologist.etl.prefect_pipelines:etl_flow` and
    `radiologist-etl/src/radiologist/etl/conf/etl.yaml` — a flow and a config
    file that were both retired by the three-stage redesign, with `source=` /
    `destination=` / `shard_root=` parameters that no current entry point
    accepts.
  - The root `README.md:86-93` shows
    `uv run --active python -m radiologist.etl.prefect_pipelines source=… ` —
    the same retired monolithic invocation.

    Both deserve their own follow-up epic (rewrite against the three
    subcommands). Deliberately excluded here so that ten behavioural fixes are
    not held hostage to a documentation rewrite.
- Any new third-party dependency, any new packaging extra, any new module.
- Rejecting an extract manifest passed to the build stage as `split_manifest=`
  (input-kind validation) — see #4's *Out of scope*.
- Re-partitioning corpora on purpose: the run-id change in #3 is a *consequence*
  of the fix, not a goal.
- Any change to `radiologist-core`. Its datamodule is the reason #2's invariant
  matters (see Data impact), but it needs no edit.

## Architecture summary

This is a defect epic, so the "architecture" is a **slicing and ownership
decision**, not a new design. Four controlling constraints:

1. **One-time run-id break.** Defects 2 and 3 both change the assign-split run
   id and, by cascade, the build run id. They are merged into a single issue
   (#3) so operators absorb exactly one re-fingerprinting, not two. This is the
   only issue in the epic that changes a run id.
2. **Single owner per file.** Three defects live in `prefect_pipelines.py`, two
   of them inside the same flow body. Rather than serialising three developers
   through one function, that file has exactly one owning issue (#6). The same
   rule gives single owners to `execution.py` (#5), `build.py` (#2),
   `identity.py` + `assign.py` (#3), `shards.py` (#4), `beam_executor.py` (#7),
   `processors.py` (#8) and the CLI `etl` group module (#9).
3. **A behaviour-preserving skeleton.** A bugfix epic cannot stub shipping
   functions with `NotImplementedError` — `build_shards` and `directory_digest`
   have live tests. #1 therefore lands the *entire changed public API surface*
   (new parameters with behaviour-preserving defaults, new dataclass fields with
   defaults, the new exception type, the new constants, the `__all__` updates)
   while changing no behaviour at all. That is what makes seven slices
   implementable in parallel against a contract that already exists on the
   branch point.
4. **Only two shared seams, both justified by a duplication that exists today.**
   - `EXTRACT_MANIFEST_PREFIX` / `EXTRACT_MANIFEST_SUFFIX` (#1 declares, #3 uses):
     the "what counts as an extract manifest" decision provably lives in two
     places right now — the assign-split folder scan and the assign-split run-id
     fingerprint — and a divergence between them *is* defect 2.
   - The `masks_root`-requires-`images_root` message constant (#8): after #8 the
     identical `ValueError` message is raised from two entry points.

   **Not introduced:** an `ExecutionBackend`/`Runner` protocol (defect 5 is a
   one-line predicate error; multi-backend execution deliberately rides
   Prefect's own `TaskRunner` plus Hydra `_target_`), and a shared failure-rate
   gate between extract and build (only one occurrence exists at epic start —
   #2 *creates* the second, so the extraction is deferred to the optional
   refactor #10 where the duplication is real).

## Data impact

**#2 fixes a manifest invariant that `radiologist-core` depends on.** Today a
build run whose images cannot be read exits 0 and writes a manifest full of
records that are `excluded=False` with `shard=None`. Downstream, in
`radiologist-core/src/radiologist/core/data/datamodule.py`:

- `train_size` / `val_size` / `test_size` (`:149`, `:157`, `:163`) each count
  `not r.excluded and r.split == "<split>"` over the records read by
  `records_reader` at `:183`. They never inspect `shard`, so every un-sharded
  record **overcounts** the split. Those sizes then drive
  `.with_epoch(...)` / `.with_length(...)` at `:313-314`, `:326-327`, `:339-340`.
- `_compute_priors` (`:202-221`) matches on the shard field at `:216` with
  `p in pathjoin(self.shard_root, r.shard)`. With `r.shard is None` that call
  reaches `PurePath.joinpath(None)` and raises `TypeError` — so the same defect
  that silently overcounts also crashes prior computation.

The invariant #2 establishes — **no record in a build manifest is both
non-excluded and shard-less** — is exactly what makes those two code paths
correct. No change to `radiologist-core` is in scope.

**#3 changes every assign-split and build run id, once.** The split manifest
stamps the assign run id on every record, and the build stage fingerprints the
split manifest's bytes, so the build id cascades. Existing artifacts are
untouched and remain readable; the next run lands under a new id. Announce
before merging #3.

## Acceptance criteria

- [ ] On an install with no optional extras, `radiologist etl extract`,
      `radiologist etl assign-split` and `radiologist etl build` all run to
      completion on the shipped default configuration and exit 0.
- [ ] A build run in which some images fail to be written into shards reports a
      non-zero failure count, and fails the run when the configured tolerance is
      exceeded.
- [ ] Every record in a build manifest that is not marked excluded carries a
      non-null shard.
- [ ] A byte-identical folder of extract manifests produces the same
      assign-split run id regardless of where the folder lives.
- [ ] Re-running assign-split twice over an unchanged input folder, with the
      output folder equal to the input folder, produces the same run id both
      times and does not grow the folder without bound.
- [ ] The shard path recorded in the build manifest resolves, relative to the
      build output directory, to the tar file that was actually written — for
      every split name, including the empty one.
- [ ] Explicitly configuring a zero-valued execution knob is either honoured or
      rejected with a clear error — never silently replaced by a default.
- [ ] A completed build run surfaces both its split-report table and its output
      link as distinct orchestration artifacts.
- [ ] After a Beam-backed dispatch completes — successfully or not — no
      intermediate scratch objects from that dispatch remain.
- [ ] Requesting mask-based features without a mask-resolution root fails
      immediately with a message naming both settings, rather than reporting
      every image as unreadable.
- [ ] The three ETL CLI subcommands emit the failure/volume fields operators
      need to gate a pipeline on.

## Dependencies

- None external. No new third-party dependency, no new extra, no new module.
- Operational: #3 changes assign-split and build run ids for every corpus.
  Announce before merging; nothing is deleted, so the change is additive and
  reversible by reverting #3.

## Build sequence

| Local # | Title                                                            | Primary files                                                                                 | Depends on | Blocks     | Phase |
| ------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ---------- | ---------- | ----- |
| 1       | Skeleton — behaviour-preserving public API contract               | `etl/models.py`, `etl/build.py`, `etl/identity.py`, `etl/__init__.py`                           | —          | 2–9        | 1     |
| 2       | Build stage surfaces and gates shard-write failures               | `etl/build.py`, `etl/conf/build.yaml`, `radiologist-etl/README.md`                              | 1          | 6, 10      | 2     |
| 3       | Assign-split input identity: extract manifests only, portable     | `etl/identity.py`, `etl/assign.py`, `radiologist-etl/README.md`                                 | 1          | —          | 2     |
| 4       | Recorded shard path matches where the tar was written             | `etl/shards.py`                                                                                 | 1          | —          | 2     |
| 5       | Local runner and `radiologist etl` work without prefect           | `etl/execution.py`, `cli/optional.py`, `radiologist-etl/README.md`, `radiologist-cli/README.md` | 1          | —          | 2     |
| 6       | Flow wiring: zero-valued knobs and distinct build artifacts       | `etl/prefect_pipelines.py`                                                                      | 1, 2       | 10         | 3     |
| 7       | Beam scratch parts are reclaimed after every dispatch             | `etl/beam_executor.py`                                                                          | 1          | —          | 2     |
| 8       | Batch processing rejects a mask root without an images root       | `etl/processors.py`, `etl/extract.py`                                                           | 1          | 10         | 2     |
| 9       | CLI result payloads report failure and volume fields              | `cli/groups/etl.py`, `radiologist-cli/README.md`                                                | 1          | —          | 2     |
| 10      | Refactor — share the failure-rate gate between extract and build  | `etl/extract.py`, `etl/build.py`                                                                | 2, 6, 8    | —          | 4     |

Paths are relative to `radiologist-etl/src/radiologist/` and
`radiologist-cli/src/radiologist/` respectively, except the two `README.md`
entries, which are package-root files.

### Concurrency

- **Phase 1 (serial, 1 dev):** #1. Small, mechanical, zero-behaviour, unblocks
  everything.
- **Phase 2 (7 devs in parallel):** #2, #3, #4, #5, #7, #8, #9. No two of these
  edit the same source file. The only shared file is
  `radiologist-etl/README.md`, touched by #2, #3, #5 and #8 at four widely
  separated anchors (build config row / assign-split config row / dependencies
  paragraph / extract `masks_root` row).
- **Phase 3 (1 dev):** #6 — sole owner of `prefect_pipelines.py`; waits on #2
  only because one of its acceptance criteria observes #2's behaviour end to end.
- **Phase 4 (optional, 1 dev):** #10 — pure refactor, no test may change.

### File ownership map (conflict-freedom proof)

| File                                              | Owning issue(s)            |
| ------------------------------------------------- | -------------------------- |
| `radiologist/etl/models.py`                       | #1                         |
| `radiologist/etl/__init__.py`                     | #1                         |
| `radiologist/etl/build.py`                        | #1 (signature), #2, #10    |
| `radiologist/etl/identity.py`                     | #1 (signature), #3         |
| `radiologist/etl/assign.py`                       | #3                         |
| `radiologist/etl/shards.py`                       | #4                         |
| `radiologist/etl/execution.py`                    | #5                         |
| `radiologist/etl/prefect_pipelines.py`            | #6                         |
| `radiologist/etl/beam_executor.py`                | #7                         |
| `radiologist/etl/processors.py`                   | #8                         |
| `radiologist/etl/extract.py`                      | #8 (message constant), #10 |
| `radiologist/etl/conf/build.yaml`                 | #2                         |
| `radiologist/cli/optional.py`                     | #5                         |
| `radiologist/cli/groups/etl.py`                   | #9                         |

Issues sharing a file are always in different phases.

## Epic-wide hard constraints

Repeated in every issue that can violate them:

1. **Execution-only knobs never enter a run-id `config` dict.**
   `max_failure_rate`, `workers` and `batch_size` must never be added to the
   mapping fed to `compute_extract_run_id` / `compute_assign_run_id` /
   `compute_build_run_id`. Doing so would silently change every existing run id.
2. **The run-id change happens exactly once, in #3.** No other issue may alter
   any run-id input or config dict.
3. **Per-unit workers collect failures as data; only stage functions decide an
   aggregate is fatal.** `process_batch` and `write_shard` keep returning
   `(path, message)` failure lists and keep never raising for a single bad
   image. The sole exception in this epic is #8's `masks_root`-requires-
   `images_root` argument guard, which describes *the call*, not an image.
4. **Python 3.10.** Every `radiologist-etl` module carries
   `from __future__ import annotations`, so `X | None` is fine there.
   `radiologist-cli` modules do **not** — use `Optional[...]` / `Dict[...]` from
   `typing` in CLI code. black at 88 columns; mypy with `no_implicit_optional`;
   commitizen commit prefixes (`fix`, `feat`, `refactor`, `test`, `docs`,
   `chore`).
5. **No new third-party dependency, no new extra, no new module.**

## Test-suite conventions (apply to every issue)

- `radiologist-etl` tests live in `radiologist-etl/radiologist_etl_tests/`;
  `radiologist-cli` tests live in `radiologist-cli/radiologist_cli_tests/`.
  Neither package uses a `tests/` directory.
- **Never mock owned code** (`radiologist.*`). Only true process boundaries may
  be stubbed: the Prefect SDK's artifact functions, the W&B SDK, HTTP, the clock.
  fsspec is a boundary but is *not* mocked in this package — tests run against
  real files under `tmp_path`.
- Optional backends are toggled with
  `monkeypatch.setattr(radiologist.etl.optional, "_PREFECT_AVAILABLE", False)`
  (or `_BEAM_AVAILABLE`, `_PREFECT_DASK_AVAILABLE`, `_PREFECT_RAY_AVAILABLE`).
  **Never `pytest.mark.skipif`.**
- Flow tests call flows through the `.fn` escape hatch (e.g.
  `extract_flow.fn(cfg, execution=plan)`) because the real Prefect engine does
  not run in this sandbox.
- The shared `radiologist_etl_tests/conftest.py` provides exactly three
  fixtures: `image_dir` (a tiny PNG corpus with `NORMAL/` and `ABNORMAL/`
  subfolders), `mask_dir`, and `minimal_records`. There is **no** manifest or
  file-listing fixture — each test file rolls its own local helper. Follow that;
  do **not** refactor the conftest as part of this epic.
