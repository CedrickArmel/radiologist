## 🐛 Recorded shard path does not match where the tar was written

**Requires:** #1 · **Blocks:** —

### Context

The shard-writing worker computes two things from the same three components
(split name, label, shard filename): the **destination** it writes the tar to,
and the **relative path** it reports back, which the build stage then stamps
into every non-excluded record's `shard` field.

It computes them two different ways. The destination is built with the project's
`pathjoin` helper, which collapses empty components. The relative path is built
by a raw `"/".join(...)`, which does not. Whenever the split name is empty, the
two disagree by a leading separator: the tar lands at
`{output_dir}/NORMAL/-normal-000000.tar` while the manifest records
`/NORMAL/-normal-000000.tar` — an absolute-looking path that, resolved against
the build output directory, points at a location that does not exist.

Every consumer that resolves a manifest's `shard` field against the build output
directory is affected, and an existing build-stage test already asserts that
resolution property
(`radiologist_etl_tests/test_build.py`,
`test_manifest_shard_location_resolves_against_output_folder`), so the invariant
is a real, pinned contract — it is simply not pinned for the empty-split case.

### Steps to reproduce

1. Obtain a manifest whose records carry an empty split. This is reachable
   today: the extract stage writes every record with `split=""`
   (`processors.py:130`), the manifest record reader defaults a missing `split`
   field to the empty string, and the build stage accepts any existing `.jsonl`
   path as its split manifest — so pointing `build split_manifest=` at an
   `extract-*.jsonl` gets there. The CLI only checks that the path exists.
2. Run the build stage against it.
3. Observed: the tar exists at `{output_dir}/{label}/-{label-lower}-000000.tar`,
   while the manifest's `shard` field reads
   `/{label}/-{label-lower}-000000.tar`.

### Expected vs actual

**Expected:** the shard path recorded for a record, resolved relative to the
build output directory, is the tar file that was actually written — for every
record, for every split name including the empty one.

**Actual:** for an empty split name the two disagree by a leading separator.

### Root cause

`radiologist-etl/src/radiologist/etl/shards.py:101-103`, in `write_shard`:

```python
shard_filename = f"{job.split}-{job.label.lower()}-{job.index:06d}.tar"
relative_path = "/".join([job.split, job.label, shard_filename])          # raw join
shard_path = fst.pathjoin(job.shard_root, job.split, job.label, shard_filename)  # collapses
```

`radiologist.utils.filesystem.pathjoin` (`radiologist-utils/src/radiologist/utils/filesystem.py:37-54`)
builds on `PurePath.joinpath`, which drops empty components. The raw
`"/".join(...)` keeps them.

**Verified divergence** (run against this repo's Python):

```
PurePosixPath('/root').joinpath('', 'NORMAL', 'x.tar')  ->  '/root/NORMAL/x.tar'   # written
'/'.join(['', 'NORMAL', 'x.tar'])                       ->  '/NORMAL/x.tar'        # recorded
```

### Behaviour to implement

Build **one** component list and feed both expressions from it, filtering empty
components before either join:

```python
shard_filename = f"{job.split}-{job.label.lower()}-{job.index:06d}.tar"
parts = [p for p in (job.split, job.label, shard_filename) if p]
relative_path = "/".join(parts)
shard_path = fst.pathjoin(job.shard_root, *parts)
```

With an empty split this yields `relative_path == "NORMAL/-normal-000000.tar"`
and a destination of `{shard_root}/NORMAL/-normal-000000.tar` — consistent. With
a non-empty split nothing changes: no component is empty, so the filter is a
no-op and both expressions produce exactly what they produce today.

**Do not derive the relative path by round-tripping through `pathjoin`.**
`pathjoin` routes its first argument through `fsspec.url_to_fs`, so
`fst.pathjoin('', 'NORMAL', 'a.tar')` absolutizes against the process's current
working directory. Verified. The one-component-list approach above is the fix.

**Do not change the shard filename pattern** (`{split}-{label-lower}-{index:06d}.tar`).
With an empty split that filename legitimately begins with `-`; changing it
would move every existing shard and change no invariant.

**Do not change the destination computation** — it is already correct.

### Acceptance criteria

- [ ] For a manifest whose records carry a non-empty split, the shard path
      recorded for each record, resolved against the build output directory, is
      an existing tar file — unchanged from today's behaviour.
- [ ] For a manifest whose records carry an empty split, the shard path recorded
      for each record, resolved against the build output directory, is an
      existing tar file.
- [ ] For a manifest whose records carry an empty split, no recorded shard path
      begins with a path separator.
- [ ] For a manifest whose records carry an empty split, the recorded shard path
      still names the label directory and the shard filename, in that order.
- [ ] The shard path a record receives is identical whether the shard-writing
      work was dispatched locally or through the Beam execution family, for the
      same manifest.
- [ ] mypy clean; pytest green

### Out of scope

- Rejecting an extract manifest supplied as the build stage's split manifest
  input. That is a *reachability path* for this defect, not the defect: after
  this fix, a build over such a manifest produces a manifest and a set of tars
  that are mutually consistent, which is the invariant that matters. Adding
  input-kind validation to the build stage or the CLI would be a behaviour
  change with its own error-code and messaging design, and is deliberately not
  bundled into a path-consistency fix.
- Renaming shard files.
- Any change to the destination path computation.

### Technical notes

- `radiologist-etl/src/radiologist/etl/shards.py` — this issue is the sole owner
  of this file for the epic, and the fix is confined to three lines inside
  `write_shard`. Nothing else in the module changes.
- No parity work is needed for the Beam execution family: `beam_executor.py`
  serialises the shard outcome to JSON (`_shard_outcome_to_json`) and
  reconstructs it verbatim, `relative_path` included, so it inherits the fix.
  The parity acceptance criterion exists to pin that.
- **Verified: no existing Beam test literal needs updating.**
  `radiologist_etl_tests/test_beam_executor.py`, in
  `test_run_shards_returns_one_outcome_per_job_in_input_order`, builds its
  expected strings as
  `f"{job.split}/{job.label}/{job.split}-{job.label.lower()}-{job.index:06d}.tar"`
  over jobs whose splits are all non-empty, so it is unaffected. Extend that
  file with an empty-split case rather than editing the existing assertion.
- The existing shard-building tests already assert that a shard outcome's
  relative path resolves against the shard root
  (`test_write_shard_relative_path_resolves_against_shard_root` in
  `radiologist_etl_tests/test_shard_building.py`); the empty-split case is an
  additive scenario alongside it, not a rewrite. No existing test in this epic's
  scope needs to change for this issue.
- The existing Beam tests run the **real** executor against Beam's direct runner
  with a real local scratch directory under `tmp_path`. Extend that pattern for
  the parity criterion; do not mock anything.
- Docs: this fix invalidates no documented behaviour. No doc changes.
