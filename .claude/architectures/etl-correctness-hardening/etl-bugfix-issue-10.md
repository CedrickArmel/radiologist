## ♻️ Refactor — share the failure-rate gate between the extract and build stages

**Requires:** #2, #6, #8 · **Blocks:** — · **Optional** — the epic is complete and correct without it

### Context

Before this epic, exactly one ETL stage aggregated per-unit failures, computed a
rate, compared it against a configured tolerance and raised: the extract stage,
at `radiologist-etl/src/radiologist/etl/extract.py:217-228`. That is why no
shared abstraction was introduced up front — a single occurrence is not a
duplication, and extracting it speculatively would have been an abstraction
built for an imagined second caller.

#2 creates the second caller. After it lands, the extract stage and the build
stage contain the same five-step sequence — aggregate failures, count, divide by
a denominator, compare against a tolerance, raise a stage-specific error whose
message follows the same template — differing only in the stage name, the
denominator's meaning (`total` images listed vs. `planned` non-excluded records)
and the exception type. #6 then wires the build stage's tolerance through the
flow layer with the same null-checking discipline the extract flow already uses.
#8 is a prerequisite only because it also edits `extract.py` (replacing an
inline message literal with an imported constant) and must not be rebased under
this refactor.

Now the duplication is real, provable and in front of a reviewer. This issue
collapses it. All observable behaviour is already implemented and tested by #2,
#6 and the pre-existing extract tests; this issue changes structure only.

### Scope

**In scope**

- Extract the shared failure-rate gate — aggregate, count, rate, compare,
  raise — into one place, parameterised by the stage name, the tolerance, the
  denominator and the exception type to raise.
- Have both the extract stage and the build stage call it.
- Keep both exception types (`ExtractionFailureError`, `BuildFailureError`) and
  both of their public exports exactly as they are. A single generic error type
  would be a behaviour change: callers already distinguish the two, and both are
  in the ETL package's `__all__`.
- Keep both raised messages byte-identical to what #2 and the current extract
  stage produce, including the `f"{p!r} ({msg})"` failure description joined
  with `"; "`. The message text is asserted by tests; if a test has to change,
  this refactor's scope is wrong.
- Keep the helper **package-private**. It is an internal implementation detail
  shared by two sibling modules, not a new public API. Do not add it to
  `__all__`; do not create a new module for it — the epic forbids new modules,
  and it fits naturally in one of the two existing ones.

**Not in scope**

- Any new behaviour, any new configuration key, any change to a failure rate's
  denominator, or to when a stage raises.
- Bug fixes. If something is found broken while refactoring, open a separate Bug
  issue.
- Unifying anything else the two stages have in common (run-id computation,
  manifest writing, mapper resolution). Those are similar in shape but differ in
  substance; only the failure gate is a genuine repetition.
- Touching the flow layer, the CLI, any config file, or any test.

### Acceptance criteria

- [ ] All existing tests pass **without modification** — no test may be changed,
      renamed, or re-parameterised to accommodate this refactor. If one must
      change, the scope is wrong: stop and reconsider.
- [ ] mypy clean; pytest green

### Technical notes

- Files involved: `radiologist-etl/src/radiologist/etl/extract.py` and
  `radiologist-etl/src/radiologist/etl/build.py`, plus wherever the shared
  helper ends up living. Both already carry `from __future__ import annotations`.
- **Hard constraint, still in force:** the stage functions remain the only place
  entitled to decide that an aggregate of failures is fatal. Per-unit workers
  (`process_batch`, `write_shard`) keep collecting failures as data. This
  refactor must not move the decision into a worker.
- **Hard constraint, still in force:** nothing this refactor touches may enter
  any run-id `config` dict.
- Do the extraction **only if it genuinely reads better** than the two inline
  versions. Two twelve-line blocks that share a shape but read clearly on their
  own are an acceptable end state; a helper with four parameters and two
  conditional branches is not an improvement. If, with both implementations in
  front of you, the shared version is not obviously simpler, **close this issue
  as won't-do and say so** — that is a legitimate outcome and the reason this
  issue is marked optional.

### Design notes

The alternative was to introduce this seam in the skeleton (#1) and have #2
implement the build stage's gate against it from the start. Rejected on the
epic's governing rule: introduce a shared seam only where the same defect
provably appears in two or more places **today**. At epic start it appeared in
one. Deferring the extraction until the second occurrence exists costs one small
follow-up commit and buys three things — #2 stays a self-contained,
independently reviewable fix; the shape of the shared helper is designed against
two real implementations rather than one real and one imagined; and if the
second implementation turns out to differ more than expected (the denominators
already do), no abstraction has to be unwound.
