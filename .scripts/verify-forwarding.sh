#!/usr/bin/env bash
# Checks SSH and GPG agent forwarding from two angles: the current shell
# (whatever context you ran this in), and a simulated Claude Code Bash tool
# environment (clean env + BASH_ENV only, no inherited SSH_AUTH_SOCK) — since
# that second context is the one that silently breaks with naive forwarding.
#
# GPG specifically needs a *real* clearsign attempt, not a cheaper proxy
# check: `gpg-connect-agent getinfo pid` false-negatives (the extra socket
# gpg-relay.sh talks to runs in "restricted mode" and forbids that verb even
# when everything is healthy), and `gpg --list-secret-keys | grep sec`
# false-positives (gpg can autostart a fresh local agent as a side effect of
# listing, which answers as if the stub exists even when nothing persisted).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SIGNINGKEY="$(git config --global --get user.signingkey || true)"

gpg_real_test() {
    if [ -z "$SIGNINGKEY" ]; then
        echo "❌ no user.signingkey configured — nothing to test"
        return 1
    fi
    if ! ls "$HOME/.gnupg/private-keys-v1.d"/*.key >/dev/null 2>&1; then
        echo "❌ no stub in private-keys-v1.d."
        return 1
    fi
    if echo "verify-forwarding-test" | gpg --batch -u "$SIGNINGKEY" --clearsign 2>/dev/null |
        grep -q 'BEGIN PGP SIGNATURE'; then
        echo "✅ real clearsign through the relay succeeded (key $SIGNINGKEY)"
        return 0
    else
        echo "❌ clearsign failed — relay or host-side agent isn't answering for real ops"
        return 1
    fi
}

echo "== SSH (current shell) =="
echo "SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-<unset>}"
if [ -S "${SSH_AUTH_SOCK:-}" ] && ssh-add -l >/dev/null 2>&1; then
    echo "✅ ssh-agent reachable, identities:"
    ssh-add -l
else
    echo "❌ no reachable ssh-agent — if this is the first attach, open one"
    echo "   interactive VS Code terminal first to prime the relay symlink"
fi

echo
echo "== GPG (current shell, real clearsign test) =="
gpg_real_test

echo
echo "== Simulating Claude Code's Bash tool (clean env, BASH_ENV only) =="
env -i HOME="$HOME" PATH="$PATH" BASH_ENV="$HOME/.bashrc" SIGNINGKEY="$SIGNINGKEY" bash -c '
    echo "SSH_AUTH_SOCK=${SSH_AUTH_SOCK:-<unset>}"
    if ssh-add -l >/dev/null 2>&1; then
        echo "✅ SSH: bash tool would reach the forwarded agent"
    else
        echo "❌ SSH: bash tool would NOT reach the forwarded agent"
    fi
    if [ -n "$SIGNINGKEY" ] && echo "verify-forwarding-test" | gpg --batch -u "$SIGNINGKEY" --clearsign 2>/dev/null | grep -q "BEGIN PGP SIGNATURE"; then
        echo "✅ GPG: bash tool can sign for real through the relay (key $SIGNINGKEY)"
    else
        echo "❌ GPG: bash tool would NOT be able to sign"
    fi
'
