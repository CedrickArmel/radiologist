## 🐛 Assign-split run ids are location-dependent and self-poisoning

**Requires:** #1 · **Blocks:** — · **Merges two defects on purpose (see Design notes)**

### ⚠️ Operational impact — announce before merging

This issue changes the assign-split run id for **every** corpus, and by cascade
the build run id, because the split manifest stamps the assign run id on every
record and the build stage fingerprints the split manifest's bytes. That is
intended and is the point of the fix.

It happens exactly **once** in this epic. Both underlying defects are corrected
in this single commit precisely so that operators re-fingerprint once, not
twice, and so the change is attributable to one revert-able commit. Existing
artifacts are untouched and remain readable; the next run simply lands under a
new id. **No other issue in this epic may alter any run-id input or config
dict.**

### Context

The assign-split stage reads a folder of extract manifests and produces one
split manifest, stamping a content-addressed run id on every record. Two
independent defects break that content-addressing, and both change the same
value.

**Defect A — the fingerprint leaks the absolute path.** The directory digest
hashes the `name` of every listing entry. fsspec returns that field as an
*absolute* path, so a byte-identical folder of manifests copied to a different
location yields a different assign-split run id. The other two stages
fingerprint their input by streaming its bytes (`content_digest`), which is
correctly location-independent; only this one leaks the path.

**Defect B — the stage reads its own output.** Both the folder scan and the
run-id fingerprint accept *any* file ending in `.jsonl`, not just the
`extract-`-prefixed manifests the extract stage writes
(`extract.py:181` writes `{destination}/extract-{run_id}.jsonl`). The split
manifest this stage produces is named `manifest-{run_id}.jsonl`
(`assign.py:144`), which also ends in `.jsonl`. When the destination folder is
the same as the manifests folder — a natural configuration — run 1 writes its
output into the folder it just read, and run 2 then fingerprints and re-reads
that output as if it were an extract manifest. Result: the run id changes on
every invocation over unchanged inputs, the folder grows without bound, and the
previous split manifest's records are fed back into the merge. Today those
re-read records survive deduplication only by accident, because `extract-` sorts
before `manifest-` and the first occurrence of a source path wins.

### Steps to reproduce

**A:**
1. Produce a folder of extract manifests and note the assign-split run id.
2. Copy that folder byte-for-byte to a different directory.
3. Run assign-split against the copy.
4. Observed: a different run id for identical content.

**B:**
1. Run assign-split with the destination folder set to the manifests folder.
2. Run it again with no other change.
3. Observed: a different run id, a second split manifest in the folder, and the
   first split manifest counted as a source manifest.

### Expected vs actual

**Expected:** the assign-split run id is a pure function of the *content* of the
folder's extract manifests. It does not move when the folder moves, and it does
not move when the stage's own previous output is sitting in the folder.

**Actual:** it moves in both cases.

### Root cause

`radiologist-etl/src/radiologist/etl/identity.py:112-116` — the digest builds
its hashed payload from entry names verbatim:

```python
pairs = sorted(
    (str(entry["name"]), entry.get("size"))
    for entry in entries
    if entry.get("type") == "file" and str(entry["name"]).endswith(suffix)
)
```

`entry["name"]` is absolute (defect A) and the filter is suffix-only (defect B,
fingerprint half). `compute_assign_run_id` (`identity.py:151-167`) calls it with
no suffix or prefix argument at all.

`radiologist-etl/src/radiologist/etl/assign.py:57` and `:91-95` — the folder
scan applies the same suffix-only filter against a private constant:

```python
_MANIFEST_SUFFIX = ".jsonl"
...
manifest_files = sorted(
    entry["name"]
    for entry in entries
    if entry.get("type") == "file" and str(entry["name"]).endswith(_MANIFEST_SUFFIX)
)
```

(defect B, scan half).

### Behaviour to implement

1. **Hash basenames, not paths.** The digest payload must be built from each
   entry's final path component paired with its size, still sorted, still
   obtained from a **single** detailed listing call. The digest must continue to
   change when a manifest is added, removed, or changes size, and must continue
   to ignore files that do not match the filter.
2. **Honour the prefix filter.** `directory_digest` already accepts a `prefix`
   parameter defaulting to `""` (added by #1, currently ignored). Give it
   meaning: an entry is included only when its **basename** starts with `prefix`
   *and* ends with `suffix`. `prefix=""` keeps today's suffix-only semantics, so
   the digest's other callers and its existing tests are unaffected.
3. **The assign-split fingerprint asks for extract manifests specifically.**
   `compute_assign_run_id` passes `prefix=EXTRACT_MANIFEST_PREFIX` and
   `suffix=EXTRACT_MANIFEST_SUFFIX` when it digests the folder. Its own
   signature does not change — the stage knows what kind of input it consumes.
4. **The folder scan uses the same two constants.** Replace `assign.py`'s
   private `_MANIFEST_SUFFIX` with the shared `EXTRACT_MANIFEST_PREFIX` /
   `EXTRACT_MANIFEST_SUFFIX` pair imported from the identity module, and filter
   on basename prefix **and** suffix. The scan and the fingerprint must select
   the identical set of files — a divergence between them *is* this defect.
5. **Empty-input error is unchanged in kind.** When no file in the folder
   matches, the stage still raises `FileNotFoundError` naming the folder (today:
   `f"No extract manifest found in {manifests_dir!r}"` at `assign.py:97`). Keep
   that shape; it may be reworded to make explicit that *extract-prefixed*
   manifests are what was looked for, but it must keep naming the folder.

### The shared seam

`EXTRACT_MANIFEST_PREFIX` / `EXTRACT_MANIFEST_SUFFIX` (public, declared and
exported by #1) are the only seam introduced here, and they are justified
precisely because the same "what counts as an extract manifest" decision
provably exists in two places today that must agree. Do not build anything
larger — no `ManifestSelector` class, no protocol, no new module. Two constants
and two call sites.

### Acceptance criteria

- [ ] Assign-split over a folder of extract manifests, and over a byte-identical
      copy of that folder at a different location, produce the same run id and
      therefore the same split manifest filename.
- [ ] Two directories whose matching files have identical basenames and sizes
      but which live at different absolute locations produce the same directory
      digest.
- [ ] Adding an extract manifest to the folder changes the run id; removing one
      changes it; changing the byte length of one changes it.
- [ ] A file in the folder that ends in `.jsonl` but does not start with
      `extract-` does not change the run id, and its records do not appear in
      the split manifest.
- [ ] Running assign-split twice in a row with the destination folder equal to
      the manifests folder produces the same run id both times, and the second
      run reports the same source-manifest count and the same record count as
      the first.
- [ ] Running assign-split twice with the destination equal to the manifests
      folder leaves exactly one split manifest in the folder, not two.
- [ ] A folder containing only non-`extract-` `.jsonl` files raises a
      file-not-found error naming the folder.
- [ ] Requesting a directory digest without specifying a prefix still hashes
      every file matching the suffix, and still issues exactly one listing call
      to the filesystem abstraction — never one stat call per entry.
- [ ] Non-manifest files (`.txt`, `.md`) alongside extract manifests are still
      ignored by both the scan and the digest — unchanged from today.
- [ ] mypy clean; pytest green

### Out of scope

- Migrating or renaming artifacts produced under the old run ids. Nothing is
  deleted; old artifacts simply stop being re-derived.
- Changing the split-manifest filename pattern (`manifest-{run_id}.jsonl`) or
  the extract-manifest filename pattern.
- Rejecting an extract manifest passed to the build stage as its split manifest
  input — that is a separate reachability path, noted in #4.

### Technical notes

- The `prefix` parameter on `directory_digest` and the two constants already
  exist (added by #1); this issue only gives them meaning. Keep the module's
  existing `FileNotFoundError`-wrapping behaviour (`identity.py:41-42`,
  `:107-110`) and its docstring promise about the single listing call.
- `radiologist-etl/src/radiologist/etl/assign.py` — after filtering, the scan
  re-attaches the protocol with `fs.unstrip_protocol(name)` (`:105`) before
  reading. Filter on the **basename** but keep reading via the full entry name,
  and keep the sort a sort of the **full names** so multi-manifest merge order
  is unchanged for existing corpora. (Merge order is load-bearing: the
  deduplication at `:107-111` keeps the first occurrence of a source path, and
  an existing test pins that "the kept duplicate is from the manifest that sorts
  first by name".)
- **Existing tests that must keep passing unmodified:**
  - `radiologist_etl_tests/test_assign_split_stage.py` writes every fixture as
    `extract-0001.jsonl`, `extract-0002.jsonl`, `extract-0003.jsonl` and always
    passes a `dest_dir` distinct from `manifests_dir`, so the prefix filter
    selects exactly the same files it does today.
  - `radiologist_etl_tests/test_identity.py`'s `directory_digest` block uses
    `a.jsonl` / `b.jsonl` with no `prefix` argument, so the `prefix=""` default
    keeps them green — including
    `test_directory_digest_uses_a_single_detailed_listing_call`, which
    monkeypatches `type(fs).ls` and asserts the call counter equals exactly 1.
    **The basename change must not introduce a second listing call or a
    per-entry stat.** Derive basenames from the entry names you already have.
  - `test_non_manifest_files_alongside_manifests_are_ignored` drops a
    `README.txt` and a `notes.md` into the folder; both remain ignored.
- Do not mock the filesystem. This package's tests exercise these paths against
  real files under `tmp_path`; the location-independence criterion is naturally
  expressed as "write the same bytes under two different `tmp_path`
  subdirectories and compare".
- **Hard constraint:** nothing about workers, batch size or failure tolerance
  may enter the assign-split run-id config dict. That dict is built at
  `assign.py:129-132` and holds only `ratios` and `run_label` — leave it exactly
  as is.
- Docs: `radiologist-etl/README.md`'s per-subcommand configuration table
  documents the `assign-split` `manifests_dir` key as "Folder of extract
  manifests to merge". Amend only that row to state that files are selected by
  the `extract-` prefix. Touch nothing else in that file — #2, #5 and #8 own
  other sections of it and are landing concurrently.

### Design notes

Defect A and defect B are separable in the code but not in their effect: each
one independently invalidates every assign-split and build run id. Shipping them
as two issues would mean two announcements, two re-runs of every downstream
pipeline, and a window in which the id changed for a reason nobody can
attribute. The pragmatic call is one commit, one announcement, one
re-fingerprint.

The alternative to the shared-constants seam was to have the assign-split stage
call the directory-digest function's own filter for its scan, unifying the two
at the function level rather than the constant level. Rejected: the scan needs
the entries themselves (to read them) while the digest needs only names and
sizes, so unifying them would mean either a second listing call — breaking the
digest's documented and test-pinned single-call contract — or a new intermediate
type. Two constants achieve the same guarantee for a fraction of the surface
area.
