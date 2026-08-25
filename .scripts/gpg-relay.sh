#!/usr/bin/env bash
# Recreates ~/.gnupg/S.gpg-agent as a socat proxy to the host's real
# gpg-agent extra socket, relayed over TCP loopback by the host-side
# launchd agent (see host-gpg-relay-install.sh, run once on the Mac).
# gpg-agent's socket lives at a fixed, homedir-relative path — no env var to
# relay, unlike SSH — so once this proxy owns that path every gpg invocation
# reaches the host's real agent, interactive terminal or Claude Code's Bash
# tool alike. Guarded against duplicate launches on container restart, same
# pattern as the egress capture in post-start.sh.
set -euo pipefail

PORT="${GPG_RELAY_PORT:-17650}"
SOCKET="$HOME/.gnupg/S.gpg-agent"

rm -f "$SOCKET"

nohup socat \
    "UNIX-LISTEN:$SOCKET,fork,reuseaddr" \
    "TCP:host.docker.internal:$PORT" \
    >>"$HOME/.gnupg/gpg-relay.log" 2>&1 </dev/null &

SOCAT_PID=$!

for _ in $(seq 1 50); do
    if [ -S "$SOCKET" ]; then
        echo "✅ GPG relay ready (pid=$SOCAT_PID): $SOCKET"
        exit 0
    fi

    if ! kill -0 "$SOCAT_PID" 2>/dev/null; then
        echo "❌ socat exited before creating $SOCKET"
        cat "$HOME/.gnupg/gpg-relay.log"
        exit 1
    fi

    sleep 0.1
done

echo "❌ Timeout waiting for GPG socket: $SOCKET"
cat "$HOME/.gnupg/gpg-relay.log"
exit 1
