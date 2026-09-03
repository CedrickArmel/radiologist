---
name: pinned-subagent-cannot-enterworktree
description: A tdd-developer subagent launched with an isolated worktree cwd cannot use EnterWorktree/ExitWorktree to hop to another worktree; use plain git worktree add + work in its own pinned dir, or checkout the needed branch inside its own worktree
metadata:
  type: feedback
---

When a subagent is spawned pinned to a worktree (isolation: "worktree", or
an explicit cwd override), `EnterWorktree`/`ExitWorktree` are not usable to
switch it into a different worktree — the tool call may report success, but
the Bash sandbox's cwd-containment check remains hard-pinned to the
original worktree, and every subsequent Bash call (even a plain `pwd`, even
with `dangerouslyDisableSandbox: true`) is rejected with "this command's
working directory resolved to the shared checkout... Refusing to run it".
`ExitWorktree` itself refuses outright with "cannot be called from a
subagent with a cwd override".

**Why:** the harness tracks a pinned worktree per subagent independently of
whatever EnterWorktree does to the git-level state; EnterWorktree changing
the branch/HEAD doesn't move the pin.

**How to apply:** if a task says "work from a different base than the one
you're pinned to" (e.g. base a fix off `main` when your worktree is
mid-feature-branch), don't try to hop worktrees. Instead:
1. Check whether your own pinned worktree's current branch already sits at
   the needed base commit (`git merge-base --is-ancestor origin/main HEAD`)
   — worktrees created fresh by the orchestrator sometimes already do.
2. If so, just `git checkout -b <new-branch>` from there.
3. If you truly need an isolated tree from a different point and can't
   reuse your own, create a *sibling* worktree with plain
   `git worktree add <path> -b <branch> <ref>` from inside your pinned
   worktree (this works fine, it's still "your" worktree issuing the git
   command) — but you cannot then `cd`/EnterWorktree into it; either treat
   it as scratch space to inspect via `git --git-dir`/`git -C`, or abandon
   the sibling-worktree idea and just work in your own pinned dir as in
   steps 1–2.
4. If you did create an unusable sibling worktree by mistake, clean it up
   with `git worktree remove <path> --force` from your own pinned worktree
   before finishing — don't leave orphaned worktree registrations behind.

See also [[feedback_worktree_shell_chaining_blocked]] and
[[feedback_worktree_venv_pth_corruption_risk]] for other pinned-worktree
Bash sandbox constraints.
