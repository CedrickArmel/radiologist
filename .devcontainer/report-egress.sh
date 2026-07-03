#!/usr/bin/env bash
# Summarize a capture-egress.sh / auto-capture log — including any
# rotate-egress.sh archives — into a unique, sorted domain list. Paste the
# output straight into init-firewall.sh's `for domain in ...` loop.
set -euo pipefail

IN="${1:-$HOME/egress-capture.log}"

{
    cat "$IN" 2>/dev/null
    for archive in "$IN".*.gz; do
        [ -f "$archive" ] && zcat "$archive"
    done
} | grep -oP '(A|AAAA)\? \K[^ ]+' \
  | sed 's/\.$//' \
  | sort -u
