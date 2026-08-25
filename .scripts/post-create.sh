#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ~/.claude, ~/.gnupg, ~/.ssh are named volumes — Docker creates them
# root-owned on first mount, so the vscode user can't write into them yet.
# ~/.claude only gets a top-level chown: ~/.claude/skills and ~/.claude/agents
# are separate read-only bind mounts from the host (see devcontainer.json),
# and `chown -R` always fails on those — by design, not a bug to swallow.
# ~/.gnupg and ~/.ssh have no such nested mounts, so they're chowned
# recursively: files created inside the volume before this fix runs (e.g. by
# a root-run process) aren't fixed by a non-recursive chown, and silently
# swallowing that failure just defers the same "Permission denied" to
# gpg/ssh later with no clue why.
sudo chown vscode:vscode ~/.claude
sudo chown -R vscode:vscode ~/.gnupg ~/.ssh
chmod 700 ~/.gnupg ~/.ssh

# --- Git identity: read name/email/signing key from the host's real --------
# .gitconfig once, write our own container-local, writable copy (the host
# file is read-only).
GIT_NAME=""
GIT_EMAIL=""
GIT_SIGNINGKEY=""
if [ -f /home/vscode/.host-gitconfig ]; then
    GIT_NAME=$(git config -f /home/vscode/.host-gitconfig --get user.name || true)
    GIT_EMAIL=$(git config -f /home/vscode/.host-gitconfig --get user.email || true)
    GIT_SIGNINGKEY=$(git config -f /home/vscode/.host-gitconfig --get user.signingkey || true)
fi
[ -n "$GIT_NAME" ] && git config --global user.name "$GIT_NAME"
[ -n "$GIT_EMAIL" ] && git config --global user.email "$GIT_EMAIL"

# --- GPG: rely entirely on the relayed host agent, no local key ------------
# gpg-agent's socket lives at a fixed, convention-based path
# ($GNUPGHOME/S.gpg-agent) — unlike SSH there's no env var to relay, so as
# long as nothing here launches a competing local agent on this homedir,
# whichever agent already answers that socket (gpg-relay.sh's socat proxy to
# the host, run from post-start.sh) is what every gpg invocation uses —
# interactive terminal or Claude Code's Bash tool alike. We deliberately
# never call `gpgconf --launch gpg-agent` here.
if [ -n "$GIT_SIGNINGKEY" ]; then
    git config --global user.signingkey "$GIT_SIGNINGKEY"
    git config --global commit.gpgsign true
    git config --global gpg.program gpg
    echo "ℹ️  git signingkey set to $GIT_SIGNINGKEY (from host ~/.gitconfig)."
fi

# ~/.gpg-stub is a read-only bind mount of ~/.gpg-stub on the host (see
# devcontainer.json) — re-imported every rebuild, so the host-side export is
# a one-time step, not a per-rebuild chore. A plain pubkey-only import isn't
# enough: gpg needs a local secret-key *stub* per GnuPG's own agent-forwarding
# convention (--export-secret-subkeys, not --export-secret-keys — it exports
# a shadow pointer to the keygrip, never actual secret material) or it won't
# even attempt to ask the agent to sign.
#
# Importing that stub still makes gpg-agent ask for a *local* protection
# passphrase for the shadow entry it's about to store (nothing to do with the
# real key's passphrase, which never leaves the host). If gpg-relay.sh's proxy
# is already the one answering ~/.gnupg/S.gpg-agent (true on any re-run after
# the container's first start), this can NEVER work through it: GnuPG's extra
# socket — what the relay forwards to — deliberately forbids enabling
# loopback pinentry over a forwarded connection, a documented
# anti-exfiltration restriction, not a bug ("setting pinentry mode 'loopback'
# failed: Forbidden"). So: evict the relay if present, do the import against
# a throwaway LOCAL agent (fine — importing a shadow stub with no real secret
# material is a purely local metadata operation), then hand the socket back.
# set -e is deliberately not allowed to kill the script here — this failing
# must not take the rest (git identity, SSH relay setup) down with it.
# "gpg --list-secret-keys | grep sec" is NOT a reliable check here: gpg can
# autostart a fresh local agent as a side effect of listing, which then
# answers HAVEKEY as if the stub exists even when nothing actually persisted
# to private-keys-v1.d/ — and the reverse also happens (a live relay through
# the extra socket can make even a plain listing fail with "problem with fast
# path key listing: Forbidden"). The only trustworthy signal is a real file
# on disk.
STUB_DIR="$HOME/.gnupg/private-keys-v1.d"
if [ -f ~/.gpg-stub/pub.asc ] && [ -f ~/.gpg-stub/sub.asc ] && ! ls "$STUB_DIR"/*.key >/dev/null 2>&1; then
    grep -q '^allow-loopback-pinentry' ~/.gnupg/gpg-agent.conf 2>/dev/null ||
        echo "allow-loopback-pinentry" >>~/.gnupg/gpg-agent.conf

    gpgconf --kill gpg-agent 2>/dev/null
    pkill -f "socat UNIX-LISTEN:$HOME/.gnupg/S.gpg-agent" 2>/dev/null || true
    rm -f ~/.gnupg/S.gpg-agent
    gpgconf --launch gpg-agent

    gpg --batch --pinentry-mode loopback --passphrase '' \
        --import ~/.gpg-stub/pub.asc ~/.gpg-stub/sub.asc 2>&1

    if ls "$STUB_DIR"/*.key >/dev/null 2>&1; then
        echo "✅ GPG stub imported from ~/.gpg-stub — 'sec#' below is correct (forwarded, not local):"
        gpg --list-secret-keys --keyid-format long
    else
        echo "⚠️  GPG stub import did not persist — commit signing won't work until this is fixed."
    fi

    gpgconf --kill gpg-agent
    rm -f ~/.gnupg/S.gpg-agent

elif [ -n "$GIT_SIGNINGKEY" ] && ls "$STUB_DIR"/*.key >/dev/null 2>&1; then
    echo "✅ GPG stub already imported (found in $STUB_DIR)."
elif [ -n "$GIT_SIGNINGKEY" ]; then
    echo "⚠️  ~/.gpg-stub is empty. One-time setup, on the host (not in the container):"
    echo "      mkdir -p ~/.gpg-stub"
    echo "      gpg --export --armor $GIT_SIGNINGKEY > ~/.gpg-stub/pub.asc"
    echo "      gpg --export-secret-subkeys --armor $GIT_SIGNINGKEY > ~/.gpg-stub/sub.asc"
    echo "   Then rebuild the container — the mount picks it up automatically from then on."
else
    echo "⚠️  No user.signingkey in host ~/.gitconfig — commit signing left unconfigured."
fi

# --- SSH: known_hosts / config only, no local key ---------------------------
ssh-keyscan -t rsa,ed25519 github.com >>~/.ssh/known_hosts 2>/dev/null
grep -q '^Host \*' ~/.ssh/config 2>/dev/null || printf 'Host *\n  StrictHostKeyChecking accept-new\n' >>~/.ssh/config
chmod 600 ~/.ssh/config ~/.ssh/known_hosts 2>/dev/null || true

# Stabilize VS Code's per-terminal forwarded SSH_AUTH_SOCK into a fixed path.
# VS Code injects a live forwarded socket only into shells it directly spawns
# as an interactive terminal — lifecycle hooks and Claude Code's Bash tool
# never see it (a clean non-interactive shell gets an empty SSH_AUTH_SOCK,
# verified directly). So: every interactive terminal captures whatever
# forwarded socket it was just given into a stable symlink; every other
# context (BASH_ENV-sourced non-interactive shells, including the Bash tool)
# points SSH_AUTH_SOCK at that stable symlink instead of the ephemeral path.
# Prepended, not appended: the base image's ~/.bashrc returns immediately for
# non-interactive shells via `case $- in *i*) ;; *) return;; esac` near the
# top, so appending here would silently never run outside a real terminal.
if ! grep -qF '# ssh-relay-forward' ~/.bashrc 2>/dev/null; then
    {
        cat <<'BASHRC_SSH_RELAY'
# ssh-relay-forward: stabilize VS Code's per-terminal forwarded SSH_AUTH_SOCK
# into a fixed path so non-interactive shells (BASH_ENV, Claude Code's Bash
# tool) can reach the same forwarded host agent interactive terminals get.
if [ -S "${SSH_AUTH_SOCK:-}" ] && [ "$SSH_AUTH_SOCK" != "$HOME/.ssh/agent.sock" ]; then
    ln -sf "$SSH_AUTH_SOCK" "$HOME/.ssh/agent.sock"
fi
export SSH_AUTH_SOCK="$HOME/.ssh/agent.sock"
BASHRC_SSH_RELAY
        cat ~/.bashrc 2>/dev/null
    } >/tmp/bashrc.new
    mv /tmp/bashrc.new ~/.bashrc
fi
