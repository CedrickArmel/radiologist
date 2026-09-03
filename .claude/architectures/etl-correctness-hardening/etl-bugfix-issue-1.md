## 🦴 ETL bugfix epic — skeleton: behaviour-preserving public API contract

**Requires:** — · **Blocks:** #2, #3, #4, #5, #6, #7, #8, #9

### Context

This epic fixes ten verified defects across `radiologist-etl` and the
`radiologist-cli` surface that fronts it. Seven of the fix slices can be written
in parallel, but only if the API surface they all compile and type-check against
already exists on the branch point. This issue lands that surface — and nothing
else.

Because this is a **bugfix** epic rather than a greenfield feature, the skeleton
cannot stub working functions with `NotImplementedError`: `build_shards` and
`directory_digest` are shipping code with a live test suite. The skeleton
discipline is therefore adapted to: **add every new name, parameter and field,
with defaults chosen so that observable behaviour is exactly what it is today.**
No control flow changes. No new logic. When this issue merges, the existing test
suite must pass completely unmodified.

Every subsequent issue in the epic implements behaviour *behind* one of the
names introduced here.

### Module layout

No new files, no new modules. Four existing modules gain declarations:

```
radiologist-etl/src/radiologist/etl/
├── models.py     # BuildResult gains two defaulted fields
├── build.py      # new BuildFailureError + reason constant; build_shards gains one defaulted param
├── identity.py   # two new naming constants; directory_digest gains one defaulted param
└── __init__.py   # re-export the four new public names
```

### Current state (verified)

```python
# radiologist-etl/src/radiologist/etl/models.py:76-94
@dataclass(frozen=True)
class BuildResult:
    run_id: str
    output_dir: str
    manifest_path: str
    report_path: str
    shard_count: int
    record_count: int
```

```python
# radiologist-etl/src/radiologist/etl/build.py:55-64
def build_shards(
    split_manifest_path: str,
    shard_root: str,
    shard_size: int = 1000,
    ratios: SplitRatios | None = None,
    workers: int | None = None,
    run_label: str | None = None,
    mapper: ShardMapper | None = None,
    storage_options: dict | None = None,
) -> BuildResult:
```

```python
# radiologist-etl/src/radiologist/etl/identity.py:90-94
def directory_digest(
    directory: str,
    suffix: str = ".jsonl",
    storage_options: dict | None = None,
) -> str:
```

### Interface contracts

All signatures below are the **post-skeleton** state. Python 3.10; every
`radiologist-etl` module already carries `from __future__ import annotations`,
so `X | None` union syntax is valid and is the existing house style in these
files.

##### `radiologist-etl/src/radiologist/etl/models.py`

```python
@dataclass(frozen=True)
class BuildResult:
    """Outcome of one build-stage run.

    Attributes:
        run_id: 16-char content-addressed id for this build run.
        output_dir: ``{shard_root}/{run_id}``.
        manifest_path: ``{output_dir}/manifest-{run_id}.jsonl``, shard field populated.
        report_path: ``{output_dir}/split-report-{run_id}.json``.
        shard_count: number of tar shards written.
        record_count: non-excluded records written into shards.
        failed: records that were planned into a shard but could not be written.
        failure_rate: ``failed / planned``, ``0.0`` when nothing was planned.
    """

    run_id: str
    output_dir: str
    manifest_path: str
    report_path: str
    shard_count: int
    record_count: int
    failed: int = 0
    failure_rate: float = 0.0
```

- The two new fields are **appended** and **defaulted**, so every existing
  construction site keeps working untouched. `BuildResult` is
  `@dataclass(frozen=True)`, so defaulted fields must come last — appending is
  not a stylistic choice, it is the only legal placement.
- Field naming and docstring shape deliberately mirror the already-correct
  `ExtractResult` in the same file (`models.py:32-52`), which carries
  `total / succeeded / failed / failure_rate / excluded`.

##### `radiologist-etl/src/radiologist/etl/build.py`

```python
class BuildFailureError(RuntimeError):
    """Raised when the share of records that could not be sharded exceeds max_failure_rate."""


SHARD_WRITE_FAILED_REASON: str = "shard_write_failed"
# contract: the exclusion reason code stamped on a record whose image could not
# be written into its tar shard. Follows the existing reason-code style used by
# the quality filters ("lung_out_of_frame", "iqr:<column>"); reason codes are
# pipe-joined when a record accumulates more than one.


def build_shards(
    split_manifest_path: str,
    shard_root: str,
    shard_size: int = 1000,
    ratios: SplitRatios | None = None,
    workers: int | None = None,
    run_label: str | None = None,
    mapper: ShardMapper | None = None,
    storage_options: dict | None = None,
    max_failure_rate: float = 0.0,
) -> BuildResult:
    # contract: behaviour unchanged by this issue. `max_failure_rate` is
    # accepted and ignored here; #2 gives it meaning. Appended last, after
    # `storage_options`, so no positional caller can break. Default 0.0 mirrors
    # `extract`'s existing default (extract.py:126).
    # raises FileNotFoundError when split_manifest_path does not exist;
    # raises ValueError when shard_size < 1.
```

- `BuildFailureError` subclasses `RuntimeError`, exactly as the existing
  `ExtractionFailureError` does (`radiologist-etl/src/radiologist/etl/extract.py:54-55`).

##### `radiologist-etl/src/radiologist/etl/identity.py`

```python
EXTRACT_MANIFEST_PREFIX: str = "extract-"
# contract: the filename prefix the extract stage writes
# ("{destination}/extract-{run_id}.jsonl", extract.py:181). The single source of
# truth for "this file is an extract manifest", shared by the assign-split
# folder scan and the assign-split run-id fingerprint so the two can never
# disagree.

EXTRACT_MANIFEST_SUFFIX: str = ".jsonl"
# contract: the filename suffix the extract stage writes.


def directory_digest(
    directory: str,
    suffix: str = ".jsonl",
    storage_options: dict | None = None,
    prefix: str = "",
) -> str:
    # contract: behaviour unchanged by this issue. `prefix` is accepted and
    # ignored here; #3 gives it meaning. Default "" == today's "match on suffix
    # alone". Appended last, after `storage_options`, so no positional caller
    # can break.
    # Must continue to obtain its listing with a SINGLE detailed listing call —
    # never one stat call per entry. A test pins this (see Technical notes).
    # raises FileNotFoundError naming the URI when the directory does not exist.
```

##### `radiologist-etl/src/radiologist/etl/__init__.py`

Add the four new public names to the imports and to `__all__`, which is
maintained in **case-insensitive alphabetical order**:

```python
from radiologist.etl.build import (
    SHARD_WRITE_FAILED_REASON,
    BuildFailureError,
    build_shards,
)
from radiologist.etl.identity import (
    EXTRACT_MANIFEST_PREFIX,
    EXTRACT_MANIFEST_SUFFIX,
    # ... existing identity imports: compute_assign_run_id, compute_build_run_id,
    #     compute_extract_run_id, config_digest, content_digest, directory_digest
)

__all__: list[str] = [
    # ... existing entries, preserving case-insensitive alphabetical order ...
    "BuildFailureError",          # after "build_shards", before "BuildResult"
    "EXTRACT_MANIFEST_PREFIX",    # after "extract_flow", before "ExtractionFailureError"
    "EXTRACT_MANIFEST_SUFFIX",
    "SHARD_WRITE_FAILED_REASON",  # after "ShardOutcome", before "SplitRatios"
]
```

The package's testing convention is that tests drive the public API exported
from `__init__.py`, so these exports are what make the parallel slices testable.

### Acceptance criteria

<!-- Behaviour-preserving by construction: the only new observable facts are the
     importability and default values of the new names. -->

- [ ] The four new names are importable from the ETL package's public namespace
      and appear in its `__all__`.
- [ ] Constructing a build result without specifying a failure count yields a
      failure count of `0` and a failure rate of `0.0`.
- [ ] Running the build stage without passing a failure tolerance produces
      exactly the same run id, output directory, manifest path, report path,
      shard count and record count as before this change, for the same inputs.
- [ ] Running the build stage twice with two different failure tolerances
      produces the same run id both times.
- [ ] Requesting a directory digest without specifying a prefix produces exactly
      the same digest as before this change, for the same directory.
- [ ] The entire existing test suite passes without a single test being
      modified, renamed or re-parameterised.
- [ ] mypy clean; pytest green

### Technical notes

- **Hard constraint:** `max_failure_rate` must never be added to the `config`
  dict that `build_shards` feeds to `compute_build_run_id`. That dict is built
  at `build.py:94-98` and holds only `shard_size`, optionally `ratios`, and
  optionally `run_label`. It is an execution-only knob; adding it would silently
  change every existing build id. Leave the dict exactly as it is. The
  two-tolerances-one-run-id acceptance criterion above pins this.
- `radiologist-etl/src/radiologist/etl/identity.py` — `EXTRACT_MANIFEST_SUFFIX`
  duplicates a value that currently lives as a private
  `_MANIFEST_SUFFIX = ".jsonl"` at `assign.py:57`. Do **not** delete or rewire
  that private constant here — #3 owns `assign.py` and will collapse it onto the
  shared pair.
- The single-listing-call contract on `directory_digest` is pinned by
  `radiologist-etl/radiologist_etl_tests/test_identity.py`, in
  `test_directory_digest_uses_a_single_detailed_listing_call`, which
  monkeypatches `type(fs).ls` on the local fsspec filesystem and asserts the
  counter equals exactly 1. Adding a defaulted parameter cannot break it, but do
  not restructure the listing.
- Docs: this issue invalidates no documented behaviour, so it updates no docs.

### Design notes

The alternative was a "big-bang" first issue that also fixed the build failure
handling (#2), on the grounds that a contract nobody implements yet is dead
weight for one merge cycle. Rejected: seven slices touching six different files
can only be handed to seven developers if none of them has to invent, and then
reconcile, the same dataclass field or exception name. One cheap, boring,
zero-risk merge buys a fully parallel Phase 2. The cost — one commit whose diff
is "declarations only" — is exactly the kind of commit that reviews in two
minutes.
