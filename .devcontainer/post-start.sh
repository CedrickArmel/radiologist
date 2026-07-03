#!/usr/bin/env bash
# Runs on every container start (including plain stop/start, not just
# creation/rebuild) — gpg-agent and ssh-agent die when the container stops,
# so they need relaunching and their passphrase caches need re-presetting
# each time, not just once at creation.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash .devcontainer/unlock-agents.sh

# Egress baseline capture — runs for the whole container lifetime instead of
# a manually-started one-off. Guarded against duplicate launches on restart.
if ! pgrep -f 'tcpdump.*udp port 53' >/dev/null 2>&1; then
    nohup sudo tcpdump -i any -n -l 'udp port 53' >> "$HOME/egress-capture.log" 2>&1 < /dev/null &
    disown || true
    echo "✅ Egress capture running in background (~/egress-capture.log) — run .devcontainer/report-egress.sh to summarize"
fi

if ! pgrep -f 'rotate-egress.sh' >/dev/null 2>&1; then
    nohup bash .devcontainer/rotate-egress.sh >/dev/null 2>&1 < /dev/null &
    disown || true
fi
