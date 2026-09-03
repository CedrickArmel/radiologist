## 🐛 Batch processing reports every image as unreadable when the images root is missing

**Requires:** #1 · **Blocks:** #10

### Context

Mask-based features are resolved by mirroring an image's path from an images
root into a masks root. A masks root is therefore meaningless without an images
root, and the extract stage guards this correctly: it raises a `ValueError`
naming both settings before it does any work
(`radiologist-etl/src/radiologist/etl/extract.py:156-160`).

But the per-batch worker `process_batch` is itself a **public entry point** — it
is exported in the ETL package's `__all__`, and it is called directly by the
Beam execution family (`beam_executor.py:199-206`) and by the flow module's
batch task (`prefect_pipelines.py:119-126`), neither of which goes through the
extract stage's guard.

Called with a masks root and no images root, the worker reaches path resolution
and dies with `TypeError: argument of type 'NoneType' is not iterable` — raised
inside `fsspec.url_to_fs(None)`. Because the worker catches **every** exception
per image and records it as a failure, the caller does not see that error; it
gets a perfectly well-formed batch outcome in which *every single image* failed
with that opaque message. A configuration mistake is reported as a corpus-wide
data problem.

This is the **one** place in this epic where a per-unit worker is allowed to
raise rather than collect a failure as data. The distinction is deliberate:
`(image path, error message)` failures describe *this image*; an invalid
argument combination describes *the call*, is identical for every image in the
batch, and can be detected before any image is touched.

### Steps to reproduce

1. Call `radiologist.etl.process_batch` with a masks root and no images root,
   over any list of image paths — for example through the Beam execution family
   or the flow module's batch task.
2. Observed: a batch outcome with zero records and one failure per input path,
   each carrying the message `argument of type 'NoneType' is not iterable`.

### Expected vs actual

**Expected:** the call fails immediately with a `ValueError` naming both
settings, before any image is read — the same error the extract stage already
raises for the same mistake.

**Actual:** every image is reported as individually unreadable, with a message
that names neither setting and points at nothing actionable.

### Root cause

`radiologist-etl/src/radiologist/etl/processors.py:163-174` — the batch worker
goes straight into its per-image loop:

```python
records: list[ManifestRecord] = []
failures: list[tuple[str, str]] = []
for path in paths:
    try:
        record = _process_one(
            path, images_root, masks_root, manifest_id, extractors, storage_options
        )
    except Exception as exc:  # noqa: BLE001 - failures are carried as data
        failures.append((path, str(exc)))
    else:
        records.append(record)
return BatchOutcome(records=records, failures=failures)
```

`_process_one` calls `_resolve_mask` (`:39-73`), which returns `None`
immediately when `masks_root is None` (`:56-57`) but otherwise reaches
`fsspec.url_to_fs(images_root, ...)` at `:61`. With `images_root=None` that
raises the `TypeError`, and the blanket `except Exception` converts it into a
per-image failure. The failing condition is therefore *exactly*
`masks_root is not None and images_root is None`.

The guard that exists only in the stage,
`radiologist-etl/src/radiologist/etl/extract.py:156-160`:

```python
if masks_root is not None and images_root is None:
    raise ValueError(
        "masks_root requires images_root to resolve the mask mirror path — "
        "both are required together"
    )
```

### Behaviour to implement

1. `process_batch` validates its arguments **before** entering the per-image
   loop: if a masks root is given and an images root is not, raise `ValueError`
   with the same message the extract stage raises today. This happens outside
   the per-image `try`, so it propagates rather than being collected.
2. The per-image behaviour is otherwise **unchanged**: a genuinely unreadable
   image is still collected as a `(path, message)` failure and never raised.
   This is an epic-wide hard constraint — per-unit workers collect failures as
   data, and this argument guard is the sole, deliberate exception because it
   describes the call rather than an image.
3. The extract stage **keeps** its own guard. It fires earlier (before the file
   listing is read), so it still gives the better failure point for the common
   path, and removing it would change where the CLI reports the error.
4. **Seam:** because the identical message is now raised from two places, move
   the message into a single module-level constant in `processors.py` and have
   `extract.py` import and use it. `extract.py` already imports `process_batch`
   from that module (`extract.py:50`), so this adds no new dependency edge and
   creates no import cycle. This is the epic's second and last shared seam, and
   it is justified by exactly two occurrences of one string that must not drift.

### Acceptance criteria

- [ ] Calling the package's public batch-processing function with a masks root
      and no images root raises a `ValueError` whose message names both the
      masks-root and the images-root settings.
- [ ] That call raises before any image is opened — it raises even when the
      supplied list of image paths is empty, and even when every supplied path
      does not exist.
- [ ] Calling the batch-processing function with both roots supplied, or with
      neither supplied, behaves exactly as it does today.
- [ ] Calling the batch-processing function with a mixture of readable and
      unreadable images, and a valid root combination, still returns a batch
      outcome carrying one record per readable image and one failure entry per
      unreadable one — it does not raise.
- [ ] Running the extract stage with a masks root and no images root raises a
      `ValueError` naming both settings — unchanged from today, and with a
      message string identical to the one the batch-processing function raises.
- [ ] Dispatching batch work through the Beam execution family with a masks root
      and no images root surfaces the invalid-argument error to the caller,
      rather than returning a batch outcome in which every image failed.
- [ ] mypy clean; pytest green

### Out of scope

- Any other argument validation in the batch worker (empty extractor list, blank
  manifest id, and so on). Only the root-pair invariant is in scope; it is the
  one that today produces a corpus-wide false negative.
- Changing how the flow module or the Beam family invoke the worker. Both
  already propagate exceptions; the fix is entirely inside the worker.
- Removing the extract stage's own guard.
- Any change to `_resolve_mask`'s behaviour when both roots are supplied.

### Technical notes

- `radiologist-etl/src/radiologist/etl/processors.py` — this issue is the sole
  owner of this file for the epic.
- `radiologist-etl/src/radiologist/etl/extract.py` — this issue touches it only
  to replace the inline message literal with the imported constant. #10 also
  touches this file, in a later phase.
- `process_batch` is a top-level, picklable function because it must survive a
  process-pool and a Beam serialisation boundary. Keep it top-level; a
  module-level string constant is safe across both.
- The constant is **package-private** (leading underscore or not, it does not go
  into `__all__`). It is an implementation detail shared by two sibling modules,
  not a new public API.
- Drive the tests through the package's public namespace
  (`from radiologist.etl import process_batch`), not the submodule — the batch
  worker is exported in the ETL package's `__all__`.
- The image fixtures for these tests already exist as shared fixtures in
  `radiologist-etl/radiologist_etl_tests/conftest.py`: `image_dir` (a tiny PNG
  corpus under `tmp_path` with `NORMAL/` and `ABNORMAL/` subfolders) and
  `mask_dir`. Use them; do not mock the filesystem. Note there is **no** shared
  manifest or file-listing fixture — each test file rolls its own helper, and
  this issue must not change that.
- The Beam criterion is expressible against the real executor and Beam's direct
  runner with a real local scratch directory under `tmp_path`, which is how the
  existing Beam tests are written.
- No existing test asserts the buggy behaviour, so no existing test needs to
  change for this issue.
- Docs: `radiologist-etl/README.md`'s per-subcommand configuration table
  documents `extract` / `masks_root` as "Segmentation mask directory" and does
  not state its dependency on `images_root`. If you add that clause, amend only
  that row — #2, #3 and #5 own other sections of the same file and may still be
  in flight.
