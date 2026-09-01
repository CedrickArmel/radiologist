---
name: etl-manifest-run-identity-181
description: issue #181 (milestone #16 epic) identity.py implementation decisions #183/#184/#185 depend on when assembling run-id inputs
metadata:
  type: project
---

Implemented issue #181 on branch `feat/181-manifest-run-identity` off
`feat/16-etl-three-stage-framework`, commit `7b3afd0`. Key facts downstream
issues (#183 extract, #184 assign-split, #185 build) need to know:

**`directory_digest` cannot use `fs.ls(path, detail=True)` naively for the
"no N+1 stat calls" AC as a black-box guarantee** — `fsspec`'s
`LocalFileSystem.ls(detail=True)` internally calls `self.info()` once per
entry (see `fsspec.implementations.local.LocalFileSystem.ls` source: it
calls `self.info(path)` for the dir itself, then `self.info(f)` per
`os.scandir` entry when `detail=True`). This is backend-internal and out of
our control. The AC's "single detailed listing call, no per-entry stat call"
is satisfied at the level of *our* code issuing exactly one `fs.ls(...,
detail=True)` call — not by literally zero internal `info()` calls inside a
particular fsspec backend. Tests that assert this must spy on `fs.ls` call
count only, not on `fs.info`, or they'll fail against `LocalFileSystem`
specifically while being backend-implementation-coupled rather than
behavior-coupled.

**Public API delivered** (all re-exported from `radiologist.etl.__init__`
and added to `__all__`):
- `content_digest(uri, storage_options=None, chunk_size=1048576) -> str` —
  streamed SHA-256, full 64-char hex.
- `config_digest(config: Mapping[str, Any]) -> str` — `json.dumps(...,
  sort_keys=True, default=str)` then SHA-256. Key order in mappings (incl.
  nested) never affects the result; **list/sequence order does** (json
  preserves list order regardless of `sort_keys`), which is exactly what
  #184 needs for split-ratio order-sensitivity — pass ratios as a list of
  `[name, value]` pairs inside the config mapping, not a dict.
- `directory_digest(directory, suffix=".jsonl", storage_options=None) ->
  str` — one `fs.ls(path, detail=True)` call, hashes sorted `(name, size)`
  pairs filtered by suffix and `type == "file"`.
- `compute_extract_run_id(file_list, config, storage_options=None) -> str`,
  `compute_assign_run_id(manifests_dir, config, storage_options=None) ->
  str`, `compute_build_run_id(split_manifest_path, config,
  storage_options=None) -> str` — each is
  `sha256(json({"stage": <name>, "input": <digest>, "config":
  config_digest(config)})).hexdigest()[:16]`. The stage name is mixed into
  the hashed payload (not a filename prefix), so identical inputs across
  stages never collide — verified in
  `test_run_ids_differ_across_stages_for_the_same_underlying_inputs`.
- **Execution settings (workers, batch size, runner family) are never
  parameters of these functions at all** — they only accept
  `config: Mapping[str, Any]`. #183/#184/#185 must assemble that mapping
  themselves from only the output-affecting subset of their Hydra config
  before calling `compute_*_run_id`; nothing in identity.py filters this for
  them.
- `records_reader(path, storage_options=None)` in `manifest.py` was already
  correct from #180 (widened, positional-compatible) — no production change
  needed here, only added regression tests
  (`test_records_reader_reads_jsonl_without_storage_options`,
  `test_records_reader_accepts_storage_options_positionally`) which passed
  immediately since the behavior pre-existed. Documented per TDD workflow
  rather than skipped, since the issue's own AC explicitly listed it.

**Environment**: same disk-starved worktree constraint as
[[project_etl_three_stage_skeleton_180]] — used the shared `radiologist`
venv's interpreter read-only (`/home/vscode/.pyenv/versions/radiologist/bin/{python,mypy,flake8,black,isort}`)
with `--confcutdir=.` for pytest. Pre-commit hooks ran fine against this
interpreter via `PATH="/home/vscode/.pyenv/versions/radiologist/bin:$PATH"
git commit`. GPG commit signing succeeded without a dedicated worktree venv.

Test file: `radiologist-etl/radiologist_etl_tests/test_identity.py` (32
tests, all behavior-anchored through the public `radiologist.etl` API).
