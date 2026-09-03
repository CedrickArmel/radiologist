---
name: worktree-scratch-clone-shares-gitdir
description: cp -r of a git-worktree directory copies the .git pointer file too, so git commands run in the copy operate on the SAME shared repo (refs, tags, HEAD) as the real worktree — never `cp -r` a worktree to experiment with git state
metadata:
  type: feedback
---

Needed a disposable git state to empirically validate `cz bump` mechanics
(see [[feedback_cz_uv_provider_cwd_relative_lockfile]]), so I ran
`cp -r <worktree-dir> /tmp/scratch` intending an isolated sandbox. In a git
**worktree**, `.git` is not a directory — it's a small text file containing
`gitdir: /path/to/main-repo/.git/worktrees/<name>`. `cp -r` copies that
pointer file verbatim, so `/tmp/scratch/.git` still points at the *same*
shared object database, refs, and per-worktree `HEAD` as the real worktree.

**What happened:** `git add -A && git commit` and `git tag` run inside
`/tmp/scratch` did not touch the real worktree's files (git only read/wrote
files under `/tmp/scratch` plus the shared `.git` objects/refs), but they DID
advance the real worktree's branch ref and create real, repo-wide tags
(`refs/tags/*` are never per-worktree). The real worktree's `git status`
afterward showed spurious modifications/deletions — an artifact of comparing
untouched real files against a HEAD that had moved forward with scratch
commits. Fixed by `git tag -d <bogus tags>` (tags are global — do this
immediately, before any other agent/workflow reads them) then
`git reset --hard <last-known-good-sha>` in the real worktree, which also
silently deleted two just-created, not-yet-committed real source files
because they had been swept into the bogus commits' tree (`reset --hard`
removes anything tracked by the discarded HEAD but absent from the target) —
had to recreate them from the conversation's own memory of their content.

**Why:** git worktrees intentionally share the object store and refs
(that's the point of `git worktree add`); the `.git` file is the only thing
distinguishing "which worktree am I" and it survives a naive file copy.

**How to apply:** never `cp -r` a worktree directory to get an "isolated"
git sandbox. Instead, either (a) use `git clone <worktree-dir> /tmp/scratch`
(a real clone gets its own object database and refs) or (b) `git worktree
add` a proper second worktree, or (c) do throwaway git experiments (extra
commits/tags to observe `cz bump`) directly against a real `git init`'d tmp
repo with just the files you need, not a copy of the live one. If you ever
do end up with polluted shared tags/branch history from this mistake: check
`git tag` and `git log --oneline -n <k>` for unfamiliar entries immediately,
`git tag -d` anything bogus before any other process/agent can observe it,
then `git reset --hard` to the last legitimate commit — and re-verify with
`git status`/`git diff` that no legitimate uncommitted work got swept away
in the process before moving on.
