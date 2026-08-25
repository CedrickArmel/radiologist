#!/usr/bin/env bash
# Background loop: keeps egress-capture.log from growing unbounded for the
# life of a long-running container. Uses copytruncate (gzip a copy aside,
# then truncate the live file to zero) rather than restarting tcpdump —
# safe because the writer holds the file open in append mode, so every
# write always seeks to current end-of-file first; truncating doesn't
# leave a gap.
set -euo pipefail

LOG="$HOME/egress-capture.log"
MAX_BYTES=$((5 * 1024 * 1024))
KEEP=5
INTERVAL=300

while true; do
    sleep "$INTERVAL"

    [ -f "$LOG" ] || continue
    size=$(stat -c '%s' "$LOG" 2>/dev/null || echo 0)
    [ "$size" -lt "$MAX_BYTES" ] && continue

    for ((i = KEEP - 1; i >= 1; i--)); do
        [ -f "$LOG.$i.gz" ] && mv "$LOG.$i.gz" "$LOG.$((i + 1)).gz"
    done
    gzip -c "$LOG" > "$LOG.1.gz"
    : > "$LOG"
done
