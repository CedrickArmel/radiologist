---
name: gpg-relay-socket-hijack
description: gpg --list-secret-keys diagnostics can start a local gpg-agent that steals the relay socket, breaking commit signing
metadata:
  type: feedback
---

Running any bare `gpg`/`gpg-agent` command (e.g. `gpg --list-secret-keys`,
`gpg-connect-agent`) as a diagnostic when a commit's GPG signing fails can
auto-start a **local** `gpg-agent --daemon`, which binds
`~/.gnupg/S.gpg-agent`. In this project's devcontainer, that socket path is
normally owned by a `socat` relay (`.scripts/gpg-relay.sh`) proxying to the
**host's real gpg-agent** over `host.docker.internal:$GPG_RELAY_PORT` — the
container itself has no usable secret key material (`sec#` = stub/offline
key in `gpg --list-secret-keys` output confirms this). Once the local daemon
grabs the socket, every subsequent `git commit` fails with `gpg: signing
failed: No secret key` even though `gpg --list-secret-keys` "sees" the key.

**Why:** the container's own gpg-agent has no private key material — signing
only works when the socket forwards to the host. Diagnostic commands that
spin up a local agent silently break that forwarding.

**How to apply:** if `git commit` fails GPG signing with "No secret key",
first check `ps aux | grep gpg-agent` — if a local
`gpg-agent --homedir /home/vscode/.gnupg` daemon is running, do **not** try
`gpg --list-secret-keys` or similar bare `gpg` invocations to debug (they
create/reuse that local daemon). Instead re-run
`.scripts/gpg-relay.sh` (or `.claude/worktrees/<name>/.scripts/gpg-relay.sh`
in a worktree) — it removes the stale `S.gpg-agent` socket and re-establishes
the socat proxy to the host — then retry the commit directly. Avoid running
gpg diagnostics at all unless the relay is confirmed dead; sandboxed
`kill`/`gpgconf --kill` calls to reclaim the socket may themselves be denied
by the auto-mode classifier, so prevention (don't probe with bare `gpg`)
beats cleanup.
