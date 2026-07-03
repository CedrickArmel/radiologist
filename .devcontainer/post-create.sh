#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# ~/.claude, ~/.gnupg, ~/.ssh are named volumes — Docker creates them
# root-owned on first mount, so the vscode user can't write into them yet.
sudo chown vscode:vscode ~/.claude ~/.gnupg ~/.ssh 2>/dev/null || true
chmod 700 ~/.gnupg ~/.ssh

# --- Git identity: read name/email from the host's real .gitconfig once, --
# write our own container-local, writable copy (the host file is read-only
# and we're about to add a container-only signing key to it).
GIT_NAME=""
GIT_EMAIL=""
if [ -f /home/vscode/.host-gitconfig ]; then
    GIT_NAME=$(git config -f /home/vscode/.host-gitconfig --get user.name || true)
    GIT_EMAIL=$(git config -f /home/vscode/.host-gitconfig --get user.email || true)
fi
[ -n "$GIT_NAME" ] && git config --global user.name "$GIT_NAME"
[ -n "$GIT_EMAIL" ] && git config --global user.email "$GIT_EMAIL"

# --- GPG: generate a container-scoped signing key on first boot ------------
# Rather than importing the host's real identity key (fragile: live-agent
# locks/sockets don't survive being copied, and it puts host key material
# inside the container), generate a dedicated key here once and persist it
# in the ~/.gnupg volume. Its public half needs adding to GitHub separately
# (Settings > SSH and GPG keys) for commits made in-container to verify.
{
    echo "pinentry-program /usr/bin/pinentry-curses"
    echo "allow-preset-passphrase"
    echo "allow-loopback-pinentry"
    echo "default-cache-ttl 34560000"
    echo "max-cache-ttl 34560000"
} > ~/.gnupg/gpg-agent.conf
chmod 600 ~/.gnupg/gpg-agent.conf

# A forwarded host GPG agent (e.g. VS Code's automatic GPG forwarding) can
# leave a proxy socket at this same path. Forwarded/extra sockets hard-disable
# loopback pinentry regardless of gpg-agent.conf, which breaks the
# non-interactive key generation below with "Forbidden". Clear any such
# socket first so gpgconf launches our own agent on this volume's homedir.
gpgconf --kill gpg-agent || true
rm -f ~/.gnupg/S.gpg-agent ~/.gnupg/S.gpg-agent.extra ~/.gnupg/S.gpg-agent.browser ~/.gnupg/S.gpg-agent.ssh
gpgconf --launch gpg-agent

if ! gpg --list-secret-keys --with-colons 2>/dev/null | grep -q '^sec'; then
    if [ -n "${GPG_PASSPHRASE:-}" ] && [ -n "$GIT_EMAIL" ]; then
        gpg --batch --pinentry-mode loopback --passphrase "$GPG_PASSPHRASE" \
            --quick-generate-key "${GIT_NAME:-radiologist devcontainer} <$GIT_EMAIL>" default default never
        echo "✅ Generated a new container-scoped GPG key for $GIT_EMAIL"
        echo "   Add this public key to GitHub (Settings > SSH and GPG keys):"
        gpg --armor --export "$GIT_EMAIL"
    else
        echo "⚠️  Skipped GPG key generation — need GPG_PASSPHRASE and a git user.email on the host"
    fi
fi

KEY_ID=$(gpg --list-secret-keys --with-colons 2>/dev/null | awk -F: '$1=="sec"{print $5; exit}')
if [ -n "$KEY_ID" ]; then
    git config --global user.signingkey "$KEY_ID"
    git config --global commit.gpgsign true
    git config --global gpg.program gpg
fi

# --- SSH: generate a container-scoped key on first boot ---------------------
# Same reasoning as GPG above — a dedicated key persisted in the ~/.ssh
# volume, rather than the host's real key. Its public half needs adding to
# GitHub separately (Settings > SSH and GPG keys) for push/pull to
# authenticate.
ssh-keyscan -t rsa,ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null
grep -q '^Host \*' ~/.ssh/config 2>/dev/null || printf 'Host *\n  StrictHostKeyChecking accept-new\n' >> ~/.ssh/config
chmod 600 ~/.ssh/config ~/.ssh/known_hosts 2>/dev/null || true

if ! ls ~/.ssh/id_* >/dev/null 2>&1; then
    if [ -n "${SSH_KEY_PASSPHRASE:-}" ]; then
        ssh-keygen -t ed25519 -N "$SSH_KEY_PASSPHRASE" \
            -C "${GIT_EMAIL:-radiologist-devcontainer}" -f ~/.ssh/id_ed25519 -q
        echo "✅ Generated a new container-scoped SSH key"
        echo "   Add this public key to GitHub (Settings > SSH and GPG keys):"
        cat ~/.ssh/id_ed25519.pub
    else
        echo "⚠️  Skipped SSH key generation — need SSH_KEY_PASSPHRASE set on the host"
    fi
fi
chmod 600 ~/.ssh/id_* 2>/dev/null || true
chmod 644 ~/.ssh/*.pub 2>/dev/null || true
grep -q 'SSH_AUTH_SOCK' ~/.bashrc || echo "export SSH_AUTH_SOCK=\$HOME/.ssh/agent.sock" >> ~/.bashrc

# Agents die whenever the container stops, so launching them + presetting
# both passphrases is shared with post-start.sh (which reruns this on every
# plain container start, not just creation).
bash "$(dirname "${BASH_SOURCE[0]}")/unlock-agents.sh"

# --- Python toolchain: pyenv + uv + project venv ----------------------------
make cpusetup

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate radiologist
make dev-install

# --- Remaining brew-parity dev tools ----------------------------------------
pipx ensurepath
pipx install commitizen
pipx install cookiecutter

npm install -g pyright
go install golang.org/x/tools/gopls@latest

echo "✅ Dev container ready. Open a new shell (or 'source ~/.bashrc') to pick up pyenv/uv."
