# Task 1 — Migrate to `uv` for package management

## Current state

- `setup.py` holds project metadata and `install_requires`
- `pyproject.toml` has only a `[build-system]` stanza (setuptools)
- `requirements.txt` has pinned dev deps (pytest, scienceplots, ipykernel, ipywidgets)
- `.venv` at repo root is Python 3.9.6 (wrong — codebase requires ≥3.11)
- `.github/workflows/tests.yml` uses `pip install -e . && pip install pytest`
- `.github/workflows/publish.yml` uses `pip install build twine && python -m build`

## Approach

1. **Rewrite `pyproject.toml`** — merge project metadata from `setup.py` into a proper
   `[project]` table. Pin `requires-python = ">=3.11"`. Move dev deps into a
   `[dependency-groups]` table (PEP 735, supported by uv ≥0.4). Keep setuptools as
   the build backend (no reason to churn that).

2. **Delete `setup.py` and `requirements.txt`** — both are superseded by the new
   `pyproject.toml`.

3. **Regenerate `.venv` with Python 3.11** via `uv sync --python 3.11`. This replaces
   the stale Python 3.9 venv.

4. **Commit `uv.lock`** — `uv lock` is run as part of `uv sync`; the lock file goes in
   the repo.

5. **Update CI workflows:**
   - `tests.yml`: use `astral-sh/setup-uv@v5`, replace pip steps with `uv sync`.
   - `publish.yml`: replace `pip install build twine` + `python -m build` + `twine upload`
     with `uv build` + `uv publish`.

6. **Update README** — replace `pip install qaravan` with `uv add qaravan`; add dev
   setup instructions (`uv sync --group dev`).

## Complications / edge cases

- `ncon_torch` and `torch` are heavy; `uv sync` will pull them. The existing `.venv`
  can be deleted first to avoid any interference.
- `scienceplots` is not on PyPI under that name in all versions — will verify it
  resolves cleanly.
- The version in `setup.py` (`0.1.57`) will be carried into `pyproject.toml` as-is;
  we are not bumping version as part of this task.
