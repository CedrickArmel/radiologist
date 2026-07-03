FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive

# pyenv build deps (https://github.com/pyenv/pyenv/wiki#suggested-build-environment)
# plus the subset of the maintainer's brew leaves that make sense system-wide in Linux.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        software-properties-common \
        ffmpeg \
        ghostscript \
        gnupg \
        pinentry-curses \
        nmap \
        tmux \
        tree \
        wget \
        shfmt \
        shellcheck \
        pipx \
        tcpdump \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.13 python3.13-venv python3.13-dev \
    && rm -rf /var/lib/apt/lists/*
