# Task 2 Report — Reorganize directory structure: legacy + new skeleton

**Status: complete. 30/30 legacy tests passing.**

## What was done

### Legacy
- Created `src/qaravan/legacy/` and copied all 22 v0.1 source files there flat
  (from `core/`, `tensorQ/`, `algebraQ/`, `applications/`).
- Subdirectory `__init__.py` files were dropped (pure re-export glue).
- Fixed all absolute cross-subpackage imports (`qaravan.core.*`, `qaravan.tensorQ.*`, etc.)
  to use intra-package relative imports (`from .module import X`).
- Created `legacy/__init__.py` that star-imports from all modules — core modules
  imported last so they win any name collisions (matching v0.1 import order).
- Created `legacy/README.md` documenting frozen status and the five known v0.1 issues.

### v0.2 scaffold
- Created `src/qaravan/core/`: `base.py`, `gates.py`, `circuits.py`, `observables.py`,
  `noise.py`, `hamiltonians.py`, `lattices.py` (all empty, one-line docstrings).
- Created `src/qaravan/backends/`: `statevector.py`, `density_matrix.py`, `mps.py`,
  `mpdo.py`, `_tn_internals.py`, `monte_carlo.py`, `clifford.py`, `matchgate.py`,
  `pauli_propagation.py` (all empty).
- Top-level `src/qaravan/__init__.py` now only sets `__version__ = "0.2.0"`.

### Tests
- Moved `test_core.py`, `test_resets.py`, `test_tn.py` into `tests/test_legacy/`.
- Updated imports from `from qaravan.core/tensorQ/applications import *`
  → `from qaravan.legacy import *`.
- Created empty `tests/test_core/` and `tests/test_backends/` with `__init__.py`.

### CLAUDE.md
- Removed the "Known sharp edges" section (moved to `legacy/README.md`).

## Non-obvious issue encountered
`initializations.py` and `gates.py` both define `random_unitary` with incompatible
signatures. In v0.1 this was accidentally resolved by test import order (`from
qaravan.core import *` ran last, so `gates.random_unitary` won). Fixed by ordering
legacy's `__init__.py` so core modules import after tensorQ modules.

Additionally, files importing `from qaravan.legacy import X` (the package-level
import) inside the legacy package itself caused circular imports. Fixed by converting
those to direct relative imports (`from .specific_module import X`).

## Things to check

1. **Verify the test suite**: `uv run pytest tests/test_legacy/ -v` should give 30 passed.
2. **Check top-level isolation**: `import qaravan; hasattr(qaravan, 'StatevectorSim')`
   should be `False`. `from qaravan.legacy import StatevectorSim` should work.
3. **Glance at `src/qaravan/legacy/README.md`** — confirm the known-issues list looks right.
4. **Glance at the new scaffold** in `src/qaravan/core/` and `src/qaravan/backends/` —
   all files should be empty with a single docstring line.
