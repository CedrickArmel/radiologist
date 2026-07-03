#!/usr/bin/env bash
# Shared by post-create.sh (after key generation) and post-start.sh (on every
# plain container start). Launches gpg-agent/ssh-agent and re-presets both
# passphrases into their caches — both agent processes die when the
# container stops, so this has to re-run on every start, not just creation.
set -euo pipefail

# --- GPG -------------------------------------------------------------------
gpgconf --kill gpg-agent || true
gpgconf --launch gpg-agent

KEY_ID=$(gpg --list-secret-keys --with-colons 2>/dev/null | awk -F: '$1=="sec"{print $5; exit}')
if [ -n "$KEY_ID" ] && [ -n "${GPG_PASSPHRASE:-}" ]; then
    PRESET_BIN=$(find /usr/lib* /usr/libexec* -name gpg-preset-passphrase 2>/dev/null | head -n1)
    if [ -n "$PRESET_BIN" ]; then
        KEYGRIPS=$(gpg --with-colons --with-keygrip --list-secret-keys | awk -F: '
            $1=="sec" || $1=="ssb" { cap=$12 }
            $1=="grp" && cap ~ /s/ { print $10 }
        ')
        for grip in $KEYGRIPS; do
            "$PRESET_BIN" --preset -P "$GPG_PASSPHRASE" "$grip"
        done
        echo "✅ GPG passphrase preset into gpg-agent cache ($(echo "$KEYGRIPS" | wc -w | tr -d ' ') signing subkey(s))"
    else
        echo "⚠️  gpg-preset-passphrase not found — commits will need an interactive passphrase prompt"
    fi
fi

# --- SSH ---------------------------------------------------------------
if ls ~/.ssh/id_* >/dev/null 2>&1; then
    eval "$(ssh-agent -a "$HOME/.ssh/agent.sock" -s)" >/dev/null

    if [ -n "${SSH_KEY_PASSPHRASE:-}" ]; then
        cat > ~/.ssh/askpass.sh <<'EOF'
#!/usr/bin/env bash
echo "$SSH_KEY_PASSPHRASE"
EOF
        chmod 700 ~/.ssh/askpass.sh
        export SSH_ASKPASS="$HOME/.ssh/askpass.sh"
        export SSH_ASKPASS_REQUIRE=force
        ADDED=0
        for key in ~/.ssh/id_*; do
            [ -f "$key" ] || continue
            case "$key" in *.pub) continue ;; esac
            ssh-add "$key" < /dev/null 2>/dev/null && ADDED=$((ADDED + 1))
        done
        echo "✅ SSH passphrase preset into ssh-agent ($ADDED key(s) added)"
    fi
fi

unset GPG_PASSPHRASE SSH_KEY_PASSPHRASE
