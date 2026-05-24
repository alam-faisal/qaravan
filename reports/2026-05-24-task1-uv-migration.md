# Task 1 Report — Migrate to `uv` for package management

**Status: complete.**

## What was done

- Rewrote `pyproject.toml` with a full `[project]` table (metadata, dependencies,
  `requires-python = ">=3.11"`). Dev deps moved into `[dependency-groups] dev`.
- Deleted `setup.py` and `requirements.txt`.
- Deleted stale Python 3.9 `.venv`; ran `uv sync --python 3.11 --group dev` to
  create a fresh Python 3.11.14 venv and generate `uv.lock` (273 KB, 82 packages).
- Updated `.github/workflows/tests.yml`: uses `astral-sh/setup-uv@v5`, runs
  `uv sync --group dev` and `uv run pytest`.
- Updated `.github/workflows/publish.yml`: uses `uv build` + `uv publish`.
- Updated `README.md` installation instructions.

## Verification steps for the user

To confirm everything is wired up correctly, open a Jupyter notebook and pick
the kernel from this project's new venv:

1. In VS Code: open the Command Palette → "Python: Select Interpreter" →
   choose the one at `.venv/bin/python` inside this repo (it will show Python 3.11).
2. In a notebook cell: `import qaravan` — should import without error.
3. Alternatively from terminal: `uv run jupyter notebook` launches a server
   using the project venv automatically.
