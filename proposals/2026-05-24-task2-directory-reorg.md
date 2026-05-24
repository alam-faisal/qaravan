# Task 2 — Reorganize directory structure: legacy + new skeleton

## Current state

`src/qaravan/` has four subdirectories: `core/` (9 files), `tensorQ/` (8 files),
`algebraQ/` (2 files), `applications/` (2 files), plus a top-level `__init__.py`.
Tests live flat in `tests/` as three files (30 tests passing).

## Plan

### Step 1 — Flatten v0.1 into `src/qaravan/legacy/`

Copy every non-`__init__.py` `.py` file from all subdirs directly into
`src/qaravan/legacy/`. Files to move (all unique basenames, no collisions):

```
base_sim.py, circuits.py, gates.py, hamiltonians.py, lattices.py, noise.py,
param_gates.py, paulis.py, skeletons.py, utils.py,       # from core/
tn.py, initializations.py, statevector_sim.py, environments.py,
density_matrix_sim.py, mps_sim.py, mpdo_sim.py, monte_carlo_sim.py,  # from tensorQ/
clifford_sim.py, matchgate_sim.py,                        # from algebraQ/
trotter.py, compilation.py                                # from applications/
```

The subdirectory `__init__.py` files are pure re-export glue and will be dropped.
The top-level `src/qaravan/__init__.py` content (`__version__ = '0.1.7'`) moves
to a `legacy/__init__.py` that also star-imports from all the flat modules
(preserving `from qaravan.legacy import *` behaviour).

### Step 2 — Fix intra-package imports in legacy files

After flattening, all absolute cross-subpackage imports break. The pattern is
mechanical — replace every `qaravan.core`, `qaravan.tensorQ`, `qaravan.algebraQ`,
`qaravan.applications` with `qaravan.legacy` in the absolute imports:

| Old | New |
|-----|-----|
| `from qaravan.core import X` | `from qaravan.legacy import X` |
| `from qaravan.core.gates import X` | `from qaravan.legacy.gates import X` |
| `from qaravan.tensorQ import X` | `from qaravan.legacy import X` |
| `from qaravan.tensorQ.statevector_sim import X` | `from qaravan.legacy.statevector_sim import X` |

Existing within-subpackage relative imports (e.g. `from .gates import Gate` in
`circuits.py`) stay as-is — they resolve correctly inside the flat `legacy/`
package.

Affected files with absolute imports to fix: `clifford_sim.py`, `matchgate_sim.py`,
`compilation.py`, `trotter.py`, `skeletons.py`, `density_matrix_sim.py`,
`environments.py`, `mps_sim.py`, `mpdo_sim.py`, `statevector_sim.py`.

### Step 3 — Write `src/qaravan/legacy/README.md`

Documents that this is frozen v0.1 code kept for reference. Includes the five
"Known sharp edges" items from CLAUDE.md (which are then removed from CLAUDE.md).

### Step 4 — Scaffold v0.2 structure

Create all the empty files from the target layout in CLAUDE.md:
`core/{base,gates,circuits,observables,noise,hamiltonians,lattices}.py`,
`backends/{statevector,density_matrix,mps,mpdo,_tn_internals,monte_carlo,clifford,matchgate,pauli_propagation}.py`,
`applications/__init__.py`.

Each empty file gets a one-line module docstring naming its role.
Top-level `src/qaravan/__init__.py` is rewritten to just `__version__ = "0.2.0"`.

### Step 5 — Reorganize tests

Move existing test files into `tests/test_legacy/` and update their imports
from `from qaravan.core import *` / `from qaravan.tensorQ import *` to
`from qaravan.legacy import *`. Add empty `tests/test_core/` and
`tests/test_backends/` dirs with `__init__.py` files.

## Complications

- **`mps_sim.py`** has `from qaravan.tensorQ import MPS, all_zero_mps, ...` — this
  imports names that come via `tensorQ/__init__.py` → `tn.py`/`initializations.py`.
  After flattening, this becomes `from qaravan.legacy import MPS, all_zero_mps, ...`
  which works because `legacy/__init__.py` star-imports from `tn.py` and
  `initializations.py`.

- **`skeletons.py`** uses `from qaravan.core.gates import Gate` (absolute despite
  being in `core/` itself). Changes to `from qaravan.legacy.gates import Gate`.

## Acceptance check

After this task: `uv run pytest tests/test_legacy/` should still give 30 passed.
