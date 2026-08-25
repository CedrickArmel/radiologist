#!/usr/bin/env bash
# Run this ON THE HOST MAC — not inside the devcontainer — once per machine
# (idempotent, safe to re-run after a reboot or a key change).
#
# Docker Desktop can't bind-mount a Unix socket across its VM boundary, which
# is why VS Code's automatic GPG forwarding shows no agent inside the
# container at all. A plain TCP loopback relay crosses that boundary fine
# (containers reach the host's 127.0.0.1 via host.docker.internal), so this
# registers a launchd agent that relays gpg-agent's "extra" socket — the
# socket GnuPG itself documents for exactly this forwarding scenario, as
# opposed to the standard socket — over TCP.
set -euo pipefail

PORT="${GPG_RELAY_PORT:-17650}"
PLIST="$HOME/Library/LaunchAgents/com.devcontainer.gpg-relay.plist"
SOCAT_BIN="$(command -v socat || true)"

if [ -z "$SOCAT_BIN" ]; then
    brew install socat
    SOCAT_BIN="$(command -v socat)"
fi

EXTRA_SOCKET="$(gpgconf --list-dirs agent-extra-socket)"
if [ ! -S "$EXTRA_SOCKET" ]; then
    gpgconf --launch gpg-agent
    EXTRA_SOCKET="$(gpgconf --list-dirs agent-extra-socket)"
fi
[ -S "$EXTRA_SOCKET" ] || {
    echo "❌ no agent-extra-socket at $EXTRA_SOCKET even after launching gpg-agent"
    exit 1
}

mkdir -p "$(dirname "$PLIST")"
cat >"$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.devcontainer.gpg-relay</string>
    <key>ProgramArguments</key>
    <array>
        <string>$SOCAT_BIN</string>
        <string>TCP-LISTEN:$PORT,bind=127.0.0.1,reuseaddr,fork</string>
        <string>UNIX-CONNECT:$EXTRA_SOCKET</string>
    </array>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><true/>
    <key>StandardErrorPath</key><string>/tmp/devcontainer-gpg-relay.err.log</string>
    <key>StandardOutPath</key><string>/tmp/devcontainer-gpg-relay.out.log</string>
</dict>
</plist>
EOF

launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "✅ GPG relay listening on 127.0.0.1:$PORT -> $EXTRA_SOCKET"
echo "   Managed by launchd (com.devcontainer.gpg-relay) — starts at login, restarts if killed."
echo "   Logs: /tmp/devcontainer-gpg-relay.{out,err}.log"
