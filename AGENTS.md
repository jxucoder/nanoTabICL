# Agent Instructions

## Package Manager

This project uses **uv** exclusively. Do not use pip, conda, or other package managers.

- `uv sync` — install/update dependencies from the lockfile
- `uv run <cmd>` — run commands in the project virtualenv
- `uv add <pkg>` — add a new dependency

## Running

```bash
uv run python -m nanotabicl.demo          # quick demo (random weights)
uv run python -m nanotabicl.nano_model    # pretrain + eval
```

## Testing

```bash
uv run python -m unittest discover -s tests
```

## Project Layout

- `src/nanotabicl/nano_model.py` — single-file architecture, pretraining, and inference
- `src/nanotabicl/demo.py` — toy classification/regression demos
- `tests/` — unit and smoke tests
- `docs/` — static website (HTML/JS/CSS)
