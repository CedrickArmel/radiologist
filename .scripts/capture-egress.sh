#!/usr/bin/env bash
# Capture DNS queries made from this devcontainer, to build an
# init-firewall.sh allowlist baseline. Run this in one terminal while you
# use the container normally (claude, make dev-install, git, pip/uv, etc.)
# in another. Stop with Ctrl+C, then run ./report-egress.sh.
set -euo pipefail

OUT="${1:-$HOME/egress-capture.log}"

echo "Capturing DNS queries to $OUT — Ctrl+C to stop, then run report-egress.sh"
sudo tcpdump -i any -n -l 'udp port 53' | tee "$OUT"
