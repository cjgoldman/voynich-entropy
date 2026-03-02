#!/bin/bash
set -e

# Git config
git config --global --add user.email "${GIT_AUTHOR_EMAIL}"
git config --global --add user.name "${GIT_AUTHOR_NAME}"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Claude CLI
curl -fsSL https://claude.ai/install.sh | bash

# ---------- Python environment ----------
cd /workspace
git submodule update --init --recursive

DEVICE="${DEVICE:-cpu}"
echo "=== Setting up Python environment (${DEVICE}) ==="

if [ "$DEVICE" = "cuda" ]; then
    # Install base deps + GPU dependency group
    uv sync --group gpu
    # Make both submodules importable (deps already resolved by root)
    uv pip install -e ./blt --no-deps
    uv pip install -e ./voynich-attack --no-deps
else
    # Install base deps only
    uv sync
    # Make voynich-attack importable
    uv pip install -e ./voynich-attack --no-deps
fi

# Register Jupyter kernel
uv run python -m ipykernel install --user --name voynich-entropy --display-name "Voynich Entropy"

echo "=== Setup complete (${DEVICE}) ==="
