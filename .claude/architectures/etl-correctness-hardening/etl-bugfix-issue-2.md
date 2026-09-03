## 🐛 Build stage silently discards shard-write failures

**Requires:** #1 · **Blocks:** #6, #10

### Context

The build stage dispatches shard-writing work units and collects one outcome per
unit. Each outcome carries both the source paths successfully written into the
tar **and** a list of `(image path, error message)` failures for images that
could not be read. The stage reads the successes and throws the failures away.

Consequences, all currently reachable with exit code 0:

- Nothing is logged when images fail.
- The build result has no failure field, so no caller can detect the problem.
- A split manifest whose images have since moved produces a run reporting
  `record_count == 0` alongside a non-zero shard count, and exits 0.
- The written manifest marks every one of those records `excluded=False` with a
  null `shard`, which breaks a manifest invariant the training datamodule
  depends on (see *Downstream impact*).

The extract stage already guards this exact class of failure correctly
(`radiologist-etl/src/radiologist/etl/extract.py:217-228`): it aggregates
per-unit failures, computes a failure rate, and raises when the rate exceeds a
configured tolerance. This issue gives the build stage full parity.

### Steps to reproduce

1. Run the assign-split stage to produce a split manifest.
2. Move or delete the source images the manifest points at.
3. Run the build stage against that manifest with shipped defaults.
4. Observed: exit code 0, a build result reporting zero records and a non-zero
   shard count, no log line naming any failure, and a manifest full of records
   that are not excluded and have no shard.

### Expected vs actual

**Expected:** the run fails loudly, naming how many records could not be
sharded, because the shipped default tolerance is zero. With a non-zero
tolerance configured, the run succeeds and the un-sharded records are marked
excluded so the manifest invariant holds.

**Actual:** the run reports success and emits a manifest that overstates the
usable training set.

### Root cause

`radiologist-etl/src/radiologist/etl/build.py:114-124` — the stage collects the
mapper's outcomes, then reads only `outcome.record_paths` (to build a
path → shard lookup) and `outcome.written` (to sum `record_count` at `:157`).
The `failures` list on each outcome is never read:

```python
outcomes = mapper(jobs) if jobs else []

path_to_shard: dict[str, str] = {}
for outcome in outcomes:
    for record_path in outcome.record_paths:
        path_to_shard[record_path] = outcome.relative_path

for record in records:
    if not record.excluded:
        record.shard = path_to_shard.get(record.path)   # -> None on failure
```

For reference, the shard-writing worker populates the failures list like this
and deliberately never raises
(`radiologist-etl/src/radiologist/etl/shards.py:116-122`):

```python
except OSError as exc:
    failures.append((record.path, str(exc)))
    continue
```

That "collect failures as data" behaviour in the worker is **correct and must
not change** — only the stage function is entitled to decide that an aggregate
is fatal.

### Downstream impact (justifies the invariant, no change in scope)

In `radiologist-core/src/radiologist/core/data/datamodule.py`:

- The manifest is read via `records_reader` at `:183`.
- `train_size` (`:149`), `val_size` (`:157`) and `test_size` (`:163`) each count
  `not r.excluded and r.split == "<split>"` and never inspect `shard`. Every
  non-excluded, shard-less record therefore **overcounts** that split. Those
  sizes drive `.with_epoch(...)` / `.with_length(...)` at `:313-314`, `:326-327`
  and `:339-340`.
- `_compute_priors` (`:202-221`) matches on the shard field at `:216` with
  `p in pathjoin(self.shard_root, r.shard)`. With `r.shard is None` that reaches
  `PurePath.joinpath(None)` and raises `TypeError`.

So the same defect both silently overcounts the training set and crashes prior
computation. **No change to `radiologist-core` is in scope for this issue** —
its correctness is restored by fixing the manifest producer.

### Behaviour to implement

Mirror the extract stage's shape:

1. Aggregate every outcome's `failures` into a single list.
2. `planned` = the number of records the stage planned into shards, i.e. the
   non-excluded records of the input manifest — the same population
   `plan_shards` groups (`shards.py:61-64` skips `record.excluded`). `failed` =
   the number of aggregated failures. `failure_rate = failed / planned`, or
   `0.0` when `planned == 0`.
3. Log a warning naming the failure count when `failed` is non-zero.
4. If `failure_rate > max_failure_rate`, raise `BuildFailureError` **before**
   writing the manifest and the split report, with a message naming the failed
   count, the planned count, the observed rate and the configured tolerance.
   Match the extract stage's message shape verbatim in structure — the existing
   template is at `extract.py:222-228`:

   ```
   build stage failed: {failed}/{planned} record(s) could not be written into a
   shard (failure rate {failure_rate:.2%} exceeds max_failure_rate
   {max_failure_rate:.2%}): {failure_desc}
   ```

   where `failure_desc` is `"; ".join(f"{p!r} ({msg})" for p, msg in failures)`,
   exactly as extract builds it.
5. Otherwise (failures tolerated): for every record whose source path appears in
   the aggregated failures, set `excluded = True` and append the reason code
   `SHARD_WRITE_FAILED_REASON` to `exclusion_reason`, pipe-joined when the record
   already carries a reason. Do this **before** the shard lookup is applied
   (`build.py:121-123`) and before the label/split counts that feed the report
   are computed (`build.py:128-135`), so the report's excluded counts and the
   manifest agree.
6. Return `failed` and `failure_rate` on the `BuildResult`.

Invariant this establishes: **no record in a build manifest is both
non-excluded and shard-less.**

### Configuration

Add to `radiologist-etl/src/radiologist/etl/conf/build.yaml`, whose current keys
are `split_manifest`, `shard_root`, `shard_size`, `split_ratios`, `workers`,
`run_label`, `storage_options` (plus a `defaults:` block selecting
`runner: local`):

```yaml
max_failure_rate: 0.0 # unshardable-record tolerance before the run fails
```

`radiologist-etl/src/radiologist/etl/conf/extract.yaml` already carries an
identically named and identically defaulted key with an inline comment; keep the
comment style consistent with its neighbours.

Wiring this key from the flow into the stage is **#6's** job, not this issue's —
`prefect_pipelines.py` has a single owner. Until #6 lands, the key is inert.

### Acceptance criteria

- [ ] Given a split manifest in which every source image is readable, the build
      stage reports a failure count of `0` and a failure rate of `0.0`, and the
      run's record count and shard count are unchanged from today.
- [ ] Given a split manifest in which some source images are unreadable and a
      tolerance of `0.0`, the build stage raises a build-failure error whose
      message names the number of records that could not be sharded, the number
      planned, the observed rate and the configured tolerance.
- [ ] The build-failure error message names each unreadable source path.
- [ ] When the build stage raises a build-failure error, no manifest and no
      split report are written for that run.
- [ ] Given a split manifest in which a minority of source images are unreadable
      and a tolerance above the resulting rate, the build stage completes and
      reports a failure count equal to the number of unreadable images and a
      failure rate equal to that count divided by the number of records planned
      into shards.
- [ ] After a completed build run that tolerated failures, every record in the
      written manifest is either marked excluded or carries a non-null shard —
      no record is both non-excluded and shard-less.
- [ ] After a completed build run that tolerated failures, each record whose
      image could not be written carries an exclusion reason containing
      `shard_write_failed`, and a record that was already excluded before the
      build retains its original reason alongside the new one.
- [ ] After a completed build run that tolerated failures, the split report's
      per-label excluded count includes the records that failed to be written.
- [ ] Given a split manifest whose records are all already excluded, the build
      stage reports a failure rate of `0.0` and does not raise, whatever the
      configured tolerance.
- [ ] Building the same split manifest twice with two different tolerance values
      produces the same run id and the same output directory both times.
- [ ] A build run whose images all fail to be written no longer reports success.
- [ ] mypy clean; pytest green

### Existing test this issue supersedes

**One existing test asserts the buggy behaviour and must be rewritten as part of
this issue.** In `radiologist-etl/radiologist_etl_tests/test_build.py`:

```python
def test_unreadable_image_is_reported_as_failure_not_written_as_shard_entry(
    tmp_path: Path,
) -> None:
    missing_path = str(tmp_path / "gone.png")
    records = [_record(missing_path, "gone.png", "NORMAL", "train")]
    manifest_path = _make_split_manifest(tmp_path, records)

    result = build_shards(manifest_path, str(tmp_path / "shards"))

    assert result.record_count == 0
    out_records = records_reader(result.manifest_path)
    assert out_records[0].shard is None
```

Its single record is unreadable, so after this fix `planned == 1`,
`failed == 1`, `failure_rate == 1.0`, and with the shipped default
`max_failure_rate=0.0` the call **raises `BuildFailureError`** instead of
returning. The test's final assertion —
`out_records[0].shard is None` on a non-excluded record — is precisely the
invariant violation this issue exists to eliminate.

Replace it with two tests that pin the corrected contract at the same public
surface: one that the default-tolerance call raises with a message naming the
counts, and one that a call with a tolerance above the observed rate returns a
result whose failure count is `1` and whose written record is marked excluded
with the `shard_write_failed` reason. This is the **only** existing test this
issue may touch.

### Out of scope

- Deleting or rolling back tar shards already written when the run is failed.
  The run id is content-addressed, so a re-run lands in the same output
  directory and overwrites them; a partial directory with no manifest is already
  recognisable as a failed run.
- Reading the new config key from the flow — that is #6.
- Any change to `radiologist-core`.

### Technical notes

- The new exception type `BuildFailureError`, the constant
  `SHARD_WRITE_FAILED_REASON`, the two new `BuildResult` fields (`failed`,
  `failure_rate`) and the `max_failure_rate` parameter on `build_shards` were
  all added by #1. Implement against them; do not re-declare them.
- **Hard constraint:** `max_failure_rate` must not enter the `config` dict at
  `build.py:94-98` that is fed to `compute_build_run_id`. That dict holds only
  `shard_size`, optionally `ratios` and optionally `run_label`. Adding an
  execution-only knob would silently change every existing build id. The
  acceptance criterion about two tolerances producing one run id pins this.
- `records_reader` returns a **list** (not a generator), so the stage already
  holds every record in memory and iterates it twice. `ManifestRecord` is a
  plain mutable `@dataclass` (unlike the frozen result types) and the stage
  already mutates `record.shard` in place at `build.py:123` — the exclusion
  marking fits naturally in that same region.
- Existing exclusion reason codes in this package are `lung_out_of_frame` and
  `iqr:<column>`, appended pipe-joined by the quality filters. Match that
  convention: an already-excluded record must end up with both reasons.
- Test the aggregate behaviour by pointing the manifest at image paths that
  genuinely do not exist and letting the real `write_shard` worker collect the
  real failures. `build_shards` accepts an injected `mapper`, which is how the
  existing build tests avoid spawning a process pool — use that, but do **not**
  mock `write_shard` or any other owned module.
- Docs: `radiologist-etl/README.md` has a per-subcommand configuration table
  (rows for `extract` and `build` around lines 165-177) whose `extract` block
  already contains a `max_failure_rate` row
  ("Unreadable-image tolerance before the run fails"). Add the matching `build`
  row next to the `shard_size` row. Change nothing else in that file — #3, #5
  and #8 own other sections of it and are landing concurrently.
