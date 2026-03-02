# Voynich Entropy

Research environment for measuring next-byte entropy of Voynich manuscript text.

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### 1. Configure your data directory

Copy the example env file and set `DATA_DIR` to a folder on your local machine where you want to store data (datasets, model outputs, etc.). This directory is mounted into the container at `/workspace/data`.

```bash
cp .devcontainer/.env.example .devcontainer/.env
# Edit .devcontainer/.env and set DATA_DIR to your local path
```

The `data/` folder is git-ignored, so anything you place there stays local and won't be committed.

### 2. Choose a devcontainer configuration

The repo provides two devcontainer configs under `.devcontainer/`:

| Config | Path | Base image | Use when |
|--------|------|------------|----------|
| **CPU** | `.devcontainer/cpu/devcontainer.json` | `python:3.12` | You don't have a GPU |
| **GPU** | `.devcontainer/gpu/devcontainer.json` | `pytorch/pytorch` + CUDA | You have an NVIDIA GPU with drivers installed |

To select one:

1. Open the repo in VS Code.
2. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).
3. Run **Dev Containers: Open Folder in Container...**
4. VS Code will detect multiple configurations and prompt you to choose **CPU** or **GPU**.

Alternatively, you can reopen an already-running container with a different config via **Dev Containers: Reopen in Container** and selecting the other option.

### 3. Environment variables

The devcontainer forwards several local environment variables into the container. Set these on your host machine before opening the container:

| Variable | Purpose |
|----------|---------|
| `GIT_AUTHOR_EMAIL` | Used for git commits inside the container |
| `GIT_AUTHOR_NAME` | Used for git commits inside the container |
| `GITHUB_TOKEN` | GitHub authentication |
| `WANDB_API_KEY` | Weights & Biases logging |

### Data management

The `DATA_DIR` path you set in `.devcontainer/.env` is bind-mounted to `/workspace/data` inside the container. Use this location for:

- Input datasets
- Model checkpoints
- Experiment outputs

Because the mount points to a directory on your host filesystem, data persists across container rebuilds. Each team member can point `DATA_DIR` to a different local path to suit their machine's storage layout.

## Python Dependency Management

### Architecture

The repo contains two git submodules (`voynich-attack` and `blt`) with incompatible dependency declarations. To make both importable in the same environment, we use a single root-level `pyproject.toml` as the **sole source of truth** for all dependency versions.

```
/workspace/
├── pyproject.toml          # Root — defines ALL dependency versions
├── .python-version         # Pins Python 3.12 for uv
├── voynich-attack/         # Submodule (team-owned)
│   ├── pyproject.toml      # dependencies = []  (empty)
│   └── voynpy/             # Importable package
└── blt/                    # Submodule (upstream)
    └── pyproject.toml      # Has its own deps — ignored via --no-deps
```

The submodules are installed as **editable packages with `--no-deps`**, which registers them as importable Python packages without pulling in their declared dependencies. The root `pyproject.toml` controls what actually gets installed.

### Dependency groups

The root `pyproject.toml` organizes dependencies into groups:

| Group | Contents | Installed when |
|-------|----------|----------------|
| **base** (`dependencies`) | numpy, pandas, matplotlib, Pillow, ipython, jupyter, ipykernel | Always |
| **gpu** | torch, xformers, and all BLT runtime deps | `uv sync --group gpu` |
| **dev** | black, isort, ruff, pytest | `uv sync --group dev` |

### What each container gets

| | CPU | GPU |
|---|-----|-----|
| Base deps | Yes | Yes |
| GPU group (torch, xformers, BLT deps) | No | Yes |
| `voynpy` importable | Yes | Yes |
| `bytelatent` (BLT) importable | No | Yes |

### Adding a new dependency

1. Edit `/workspace/pyproject.toml` — add the package to `dependencies` (if needed everywhere) or the `gpu` group (if GPU-only).
2. Run `uv lock` to regenerate the lock file.
3. Run `uv sync` (CPU) or `uv sync --group gpu` (GPU) to install.
4. Commit `pyproject.toml` and `uv.lock`.

### When the BLT submodule updates

1. Compare BLT's `pyproject.toml` dependency list with the root's `gpu` group.
2. Copy any new or changed entries into the root `pyproject.toml`.
3. Run `uv lock` to regenerate the lock file.
4. Commit the updated `pyproject.toml` and `uv.lock`.

## Running the Devcontainers

### Quick start

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url>
cd voynich-entropy

# 2. Set up local env file
cp .devcontainer/.env.example .devcontainer/.env
# Edit .devcontainer/.env and set DATA_DIR

# 3. Open in VS Code and select CPU or GPU config
code .
# Ctrl+Shift+P → "Dev Containers: Open Folder in Container..."
```

### What happens on container start

The `post-create.sh` script runs automatically and performs:

1. Configures git identity from `GIT_AUTHOR_EMAIL` / `GIT_AUTHOR_NAME`.
2. Installs [uv](https://docs.astral.sh/uv/) if not already present.
3. Installs the Claude CLI.
4. Initializes git submodules (`voynich-attack`, `blt`).
5. Installs Python dependencies based on the `DEVICE` environment variable:
   - **CPU** (`DEVICE=cpu`): `uv sync` + editable install of `voynich-attack`
   - **GPU** (`DEVICE=cuda`): `uv sync --group gpu` + editable install of both `voynich-attack` and `blt`
6. Registers a Jupyter kernel named **"Voynich Entropy"**.

### Using Jupyter

After the container starts, open any `.ipynb` file and select the **"Voynich Entropy"** kernel. Example usage:

```python
# CPU or GPU — voynich-attack is always available
from voynpy.corpora import vms, latin, german
from voynpy.reftext import RefText

# GPU only — BLT requires the gpu dependency group
from bytelatent.model.blt import ByteLatentTransformer
import torch
```

### Rebuilding the container

If you change `pyproject.toml` or `post-create.sh`, rebuild the container:

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`).
2. Run **Dev Containers: Rebuild Container**.

Alternatively, from within a running container you can re-run the dependency install manually:

```bash
# CPU
uv sync && uv pip install -e ./voynich-attack --no-deps

# GPU
uv sync --group gpu && uv pip install -e ./blt --no-deps && uv pip install -e ./voynich-attack --no-deps
```
