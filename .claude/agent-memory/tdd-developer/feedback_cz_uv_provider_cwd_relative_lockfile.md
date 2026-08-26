---
name: cz-uv-provider-cwd-relative-lockfile
description: commitizen's uv version_provider resolves both pyproject.toml and uv.lock relative to process cwd, with no override — breaks bumping a workspace member from its own directory
metadata:
  type: project
---

`commitizen`'s `uv` version provider (`UvProvider`/`TomlProvider` in
`commitizen/providers/uv_provider.py`) hardcodes `Path() / "pyproject.toml"`
and `Path() / "uv.lock"` — both resolved against the process's current working
directory, with no `--config`-relative or project-root-aware resolution.

**Why this matters for a uv workspace with per-member `[tool.commitizen]`
blocks:** to bump a single member's version (not the root), `cz bump` must run
with `cwd` == that member's directory, so it edits *that* member's
`pyproject.toml`. But `uv.lock` only exists at the repository root — so
`cz bump --files-only` crashes with `FileNotFoundError: uv.lock` the moment
cwd is a member directory. Confirmed empirically (radiologist repo,
commitizen 4.x, `version_provider = "uv"`): running `cz bump` from repo root
edits the root's `pyproject.toml`/`uv.lock` fine; running it from
`radiologist-core/` throws until `uv.lock` is made resolvable from that cwd.

**How to apply:** before running `cz bump` in a member directory, symlink
the root lockfile into that directory (`ln -s "$ROOT/uv.lock"
"$MEMBER_DIR/uv.lock"`) — the provider follows the symlink and writes through
to the real root file. Remove the symlink afterward so it doesn't get treated
as a real changed/tracked file. No such workaround is needed when bumping the
workspace root itself (cwd already has both files directly). This is a
`commitizen` behavior, not project code — check `commitizen`'s installed
version's provider source if this stops matching, since it could change
between releases. See [[feedback_worktree_scratch_clone_shares_gitdir]] for
an unrelated but co-discovered worktree danger from the same investigation.
