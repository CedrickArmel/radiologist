#!/usr/bin/env bash
# Runs on every container start (including plain stop/start, not just
# creation/rebuild). No local ssh-agent to relaunch — it relies on VS Code's
# per-terminal forwarding (see post-create.sh's bashrc relay). gpg-agent has
# no such per-terminal forwarding available (Docker Desktop can't bind-mount
# Unix sockets across its VM boundary), so its TCP-relayed proxy is launched
# here instead — see gpg-relay.sh and host-gpg-relay-install.sh.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Egress baseline capture — runs for the whole container lifetime instead of
# a manually-started one-off. Guarded against duplicate launches on restart.
if ! pgrep -f 'tcpdump.*udp port 53' >/dev/null 2>&1; then
    nohup sudo tcpdump -i any -n -l 'udp port 53' >> "$HOME/egress-capture.log" 2>&1 < /dev/null &
    disown || true
    echo "✅ Egress capture running in background (~/egress-capture.log)."
fi

if ! pgrep -f 'rotate-egress.sh' >/dev/null 2>&1; then
    nohup bash .scripts/rotate-egress.sh >/dev/null 2>&1 < /dev/null &
    disown || true
fi
