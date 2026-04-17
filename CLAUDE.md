# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research environment for measuring next-byte entropy of Voynich manuscript text using Meta's Byte Latent Transformer (BLT) entropy model. Runs in VS Code devcontainers (CPU or GPU variants).

## Commands

```bash
# Install deps (CPU)
uv sync && uv pip install -e ./voynich-attack --no-deps

# Install deps (GPU — adds torch, xformers, BLT)
uv sync --group gpu && uv pip install -e ./blt --no-deps && uv pip install -e ./voynich-attack --no-deps

# Dev tools (black, isort, ruff, pytest)
uv sync --group dev

# Run tests
cd /workspace && uv run pytest tests/

# Run a single test
uv run pytest tests/test_hf_data_samp.py -k test_name

# Format
uv run black src/ tests/
uv run isort src/ tests/

# Lint
uv run ruff check src/ tests/

# Run BLT example (GPU only, from /workspace/src/)
cd /workspace/src && uv run python basic_run/blt_example.py
```

## Dependency Management

Single root `pyproject.toml` is the sole source of truth for all dependency versions. The two git submodules (`voynich-attack` and `blt`/`bytelatent`) **must always be installed with `--no-deps`** so their own dependency declarations are ignored — the root pyproject.toml controls what actually gets installed. Forgetting `--no-deps` on `blt` is a common mistake that pulls in conflicting dependencies. When adding dependencies, edit the root `pyproject.toml`, then `uv lock` and `uv sync`.

## Architecture

### Source layout (`src/`)

Scripts in `src/` are run from the `src/` directory and import each other as top-level modules (not a package). They also import from the two submodule packages:

- **`voynpy`** (from `voynich-attack/`) — Voynich corpus data: `RefText` class, pre-built corpus objects (`vms`, `vms_unicode`, `latin`, `german`, etc.)
- **`bytelatent`** (from `blt/`) — Meta's BLT model (GPU only)

### Pipeline flow

1. **`vms_uprep.py`** — Converts Voynich Unicode DataFrame into annotated byte sequences. Produces `AnnotatedChunk` objects with manuscript provenance (folio/par/line/token) preserved per character.
2. **`vms_annot.py`** — Dataclasses (`GlyphAnnotation`, `AnnotatedLine`, `AnnotatedChunk`, `SegmentKind`) that carry provenance metadata through the pipeline.
3. **`entropy_proc.py`** — Attaches per-byte entropy values from BLT model output onto `AnnotatedChunk` annotations.
4. **`voy_entropy_display.py`** — Jupyter notebook rendering: HTML entropy tables with Voynich font glyphs, color-coded entropy bars, and plots.
5. **`voy_font.py`** — Loads Voynich Unicode font from `voynich_fonts/` for notebook display.
6. **`hf_data_samp.py`** — Streaming sampler for pulling text from HuggingFace datasets for comparative entropy analysis.
7. **`basic_run/blt_example.py`** — GPU script: loads BLT entropy model, runs inference, displays results with Rich terminal output.

### Tests

Tests live in `tests/` and use `sys.path.insert(0, "src")` to import source modules. Run from workspace root.

### Specs

Feature specs live in `specs/`. These describe planned or implemented features.

### Data

`data/` is gitignored and bind-mounted from the host via `DATA_DIR` in `.devcontainer/.env`. Used for datasets, model checkpoints, and experiment outputs.
