#!/usr/bin/env bash
# Standalone, rerunnable stub (re)import — consolidates what post-create.sh
# does into one script with a real checkpoint after each step, since the
# obvious check ("gpg --list-secret-keys | grep sec") is an unreliable
# false-positive machine: gpg can silently autostart a fresh local agent as a
# side effect of listing, which answers HAVEKEY correctly for a freshly
# imported stub even when nothing actually persisted to disk. The only
# trustworthy signal that the stub persisted is a real file under
# private-keys-v1.d/ — that's what every check here uses.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SOCKET="$HOME/.gnupg/S.gpg-agent"
STUB_DIR="$HOME/.gnupg/private-keys-v1.d"

echo "== 1. Stop anything on the socket (relay or stray local agent) =="
gpgconf --kill gpg-agent 2>/dev/null
pkill -f "socat UNIX-LISTEN:$SOCKET" 2>/dev/null
rm -f "$SOCKET"
sleep 1
echo "   done."

echo
echo "== 2. Launch a genuine local agent for the import =="
grep -q '^allow-loopback-pinentry' ~/.gnupg/gpg-agent.conf 2>/dev/null ||
    echo "allow-loopback-pinentry" >>~/.gnupg/gpg-agent.conf
gpgconf --launch gpg-agent
gpg-connect-agent 'getinfo pid' /bye || {
    echo "❌ can't even reach a freshly launched local agent — stopping here."
    exit 1
}

echo
echo "== 3. Import =="
if [ ! -f ~/.gpg-stub/pub.asc ] || [ ! -f ~/.gpg-stub/sub.asc ]; then
    echo "❌ ~/.gpg-stub/{pub,sub}.asc missing — nothing to import."
    exit 1
fi
gpg --batch --pinentry-mode loopback --passphrase '' \
    --import ~/.gpg-stub/pub.asc ~/.gpg-stub/sub.asc

echo
echo "== 4. Verify it actually persisted to disk =="
ls -la "$STUB_DIR"
if ! ls "$STUB_DIR"/*.key >/dev/null 2>&1; then
    echo "❌ still nothing in $STUB_DIR — import did not persist. Stopping"
    echo "   before touching the relay so the error above stays visible."
    exit 1
fi
echo "✅ stub file(s) present."
gpg --list-secret-keys --keyid-format long

echo
echo "== 5. Tear down the local agent, hand the socket to the relay =="
gpgconf --kill gpg-agent
rm -f "$SOCKET"
sleep 1
bash .scripts/gpg-relay.sh
sleep 2

echo
echo "== 6. Real end-to-end test through the relay =="
gpg-connect-agent 'getinfo pid' /bye
