## 🐛 Beam scratch parts are never reclaimed

**Requires:** #1 · **Blocks:** —

### Context

A Beam pipeline cannot return a collection to its driver, so the Beam execution
family has each work unit write its outcome as a JSON object under a
per-dispatch scratch prefix, then reads those objects back in input order. The
prefix is minted fresh per dispatch from a UUID, so two runs can never read each
other's parts — that uniqueness property is documented and is correct.

What is not implemented is reclamation. Nothing ever deletes the prefix. Every
dispatch leaves one JSON object per work unit behind, forever. A one-million-image
extract at the default batch size of 64 leaves roughly **15,600 objects** — and
for a Dataflow run the scratch directory must be a shared remote URI
(`beam_executor.py:261-267` enforces this), so those objects accumulate in
object storage, at cost, indefinitely. Repeated runs multiply it.

### Steps to reproduce

1. Configure the Beam execution family with a scratch directory
   (`runner=beam_direct runner.beam.parts_dir=...`).
2. Run the extract stage over any corpus.
3. Observed: the run succeeds, and the scratch directory contains one JSON
   object per dispatched batch, under a UUID-named prefix, which nothing ever
   removes.
4. Run the same stage again. Observed: a second full set of objects under a
   second prefix.

### Expected vs actual

**Expected:** the scratch prefix is an implementation detail of a single
dispatch. Once the dispatch is over — whether it succeeded or failed — the
prefix is gone.

**Actual:** it persists forever, once per dispatch, for the life of the storage
location.

### Root cause

`radiologist-etl/src/radiologist/etl/beam_executor.py:352-354` mints a
per-dispatch prefix:

```python
def _run_prefix(self, kind: str) -> str:
    """A parts prefix unique to this dispatch, so runs never read each other's."""
    return f"{self.parts_dir.rstrip('/')}/{kind}-{uuid.uuid4().hex}"
```

Both dispatch entry points follow the same shape — mint a prefix, run the
pipeline, then read exactly `len(units)` parts back and reconstruct the
outcomes. Neither has a cleanup step. `run_batches` is at `:273-316`,
`run_shards` at `:318-350`; both currently end with

```python
parts_prefix = self._run_prefix(kind)
self._run_pipeline(...)
return [ _..._from_json(payload)
         for payload in _read_parts(parts_prefix, len(units), self.storage_options) ]
```

`_read_parts` (`:126-148`) deliberately reads only this run's own prefix, which
is what makes deleting that prefix safe: nothing outside this dispatch can be
referencing it.

### Behaviour to implement

1. Both dispatch entry points (`run_batches` and `run_shards`) delete their own
   scratch prefix, recursively, once the dispatch is over.
2. **Deletion happens on success AND on failure** — including when the pipeline
   itself fails, and including when the read-back finds a part missing. Structure
   it as a `try` / `finally` around the run-and-read-back, so the payloads are
   materialised inside the `try` and the outcomes are reconstructed after the
   `finally`:

   ```python
   parts_prefix = self._run_prefix("batches")
   try:
       self._run_pipeline(...)
       payloads = _read_parts(parts_prefix, len(units), self.storage_options)
   finally:
       self._reclaim(parts_prefix)
   return [_batch_outcome_from_json(p) for p in payloads]
   ```

   The originating error must still propagate unchanged; cleanup must never mask
   it.
3. Deletion is **best-effort**: if the scratch location cannot be removed (a
   permissions problem, an already-vanished prefix, a backend that does not
   implement recursive removal), log a **warning naming the prefix** and carry
   on. A failure to clean up must never turn a successful dispatch into a failed
   one, and must never replace the exception that is already propagating.
4. The scratch *directory itself* — the configured `parts_dir` — is never
   deleted. Only the per-dispatch prefix underneath it.
5. Nothing about the prefix-naming scheme changes; uniqueness stays exactly as
   documented.

### Acceptance criteria

- [ ] After a successful batch-processing dispatch through the Beam execution
      family, the configured scratch directory contains no files, and the
      dispatch's outcomes are returned in input order with their records and
      failures intact.
- [ ] After a successful shard-writing dispatch through the Beam execution
      family, the configured scratch directory contains no files, and the
      dispatch's outcomes are returned in input order.
- [ ] After two successive dispatches through the same executor and the same
      scratch directory, the scratch directory contains no files.
- [ ] When a dispatch fails because an expected outcome was not written, the
      originating error still propagates with its original message, and the
      scratch directory is left with no files.
- [ ] When the scratch location cannot be removed, a dispatch that otherwise
      succeeded still returns its outcomes rather than raising, and emits a
      warning naming the prefix that could not be removed.
- [ ] A dispatch of zero work units returns an empty result without creating a
      scratch prefix and without attempting any cleanup.
- [ ] The outcomes a Beam dispatch produces are identical to those the local
      dispatch path produces for the same input — unchanged from today.
- [ ] mypy clean; pytest green

### Out of scope

- Reclaiming scratch objects left behind by dispatches that ran *before* this
  fix. Those are addressed operationally by deleting the scratch directory's
  contents once; no migration code.
- Any retention or debugging knob to keep parts around (see Design notes).
- Changing how outcomes are serialised or read back, or the prefix-naming
  scheme.

### Technical notes

- `radiologist-etl/src/radiologist/etl/beam_executor.py` — this issue is the
  sole owner of this file for the epic.
- Both entry points already short-circuit on an empty unit list before minting a
  prefix (`:298-299` and `:336-337`), which is why the zero-unit criterion holds
  without extra work — but it must still hold after the change. Put the
  `try`/`finally` *after* the short-circuit, not around it.
- The module resolves paths through `fsspec.url_to_fs`; recursive removal is
  `fs.rm(path, recursive=True)`. Guard for the backend not implementing it
  (`NotImplementedError`) and for the path already being gone
  (`FileNotFoundError`), plus the general failure case.
- The "expected outcome was not written" failure is `_read_parts`'s own
  `RuntimeError` at `:142-145`, whose message is
  `f"Beam pipeline reported success but wrote no part for unit {index}: expected {...!r}"`.
  That is the error the corresponding criterion must observe propagating
  unchanged.
- fsspec is a true process boundary, but do **not** mock it here: the existing
  Beam tests in `radiologist-etl/radiologist_etl_tests/test_beam_executor.py`
  run the **real** executor against Beam's direct runner with a real local
  scratch directory under `tmp_path`, and every criterion above is directly
  observable that way. Extend that pattern.
- To exercise the "cannot be removed" criterion, make the removal genuinely fail
  at the filesystem level — for example by making the scratch prefix's parent
  directory unwritable after the pipeline has run — rather than by patching
  owned code.
- The Beam availability sentinel is read at access time via
  `optional._BEAM_AVAILABLE` (`beam_executor.py:55`, `:256`), so tests toggle it
  with `monkeypatch.setattr(radiologist.etl.optional, "_BEAM_AVAILABLE", False)`.
  Never `pytest.mark.skipif`.
- Docs: this fix invalidates no documented behaviour — the prefix's uniqueness
  is documented, its permanence is not. No doc changes.

### Design notes

**Reclaim on failure as well as on success — settled user decision, do not
re-litigate.** Two competing designs proposed keeping the parts prefix on
failure so it could be inspected post-mortem. That was explicitly rejected: a
failed large Dataflow run is *precisely* the case that leaks ~15,600 objects,
and precisely the case nobody comes back to sweep. Post-mortem value is
theoretical — the parts are per-unit JSON serialisations of outcomes the
exception message already summarises — while the leak is certain, recurring and
billed. The `finally` covers both paths.

For the same reason there is no opt-in "keep parts" flag. The prefix is a UUID
unrecoverable to anyone but the dispatch that made it, so retaining it has no
consumer, and adding an opt-in flag for an in-package capability is against this
project's conventions.
