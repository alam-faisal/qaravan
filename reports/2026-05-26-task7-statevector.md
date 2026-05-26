# Task 7 Report — `Statevector` and `StatevectorSimulator`

**Date:** 2026-05-26  
**Status:** Success

---

## Summary

Implemented `backends/statevector.py` exactly as proposed, with one design correction noted below. Also applied two minor prerequisite changes to `core/`.

### Prerequisite changes

| File | Change |
|---|---|
| `core/base.py` | Renamed `sample_and_collapse` → `measure_and_collapse`; flipped return order to `(State, str)` |
| `core/observables.py` | Added `num_sites` attribute to `PauliString` and `PauliSum` |
| `tests/test_core/test_base.py`, `test_gates.py` | Updated `MinimalState` / `_MinState` stubs to match the rename |

### New public API in `backends/statevector.py`

**Module-level primitives (ported verbatim from `legacy/statevector_sim.py`):**

```python
def op_action(op: np.ndarray, indices: list[int], sv: np.ndarray, local_dim: int = 2) -> np.ndarray
def partial_overlap(sv1: np.ndarray, sv2: np.ndarray, local_dim: int = 2, skip: list[int] | None = None) -> np.ndarray
```

`op_action` applies a gate at (possibly non-contiguous, out-of-order) site indices via sort+permute+ncon.  
`partial_overlap` computes the partial inner product contracting all sites not in `skip`; `skip=[]` gives the full overlap as a `(1,1)` array.

**Private helpers:**

```python
def _locs_to_indices(locs: list[int], n: int) -> tuple[list[int], list[int]]
def _expectation_pauli_string(tensor, obs: PauliString, local_dim=2) -> complex
def _expectation_local_op(tensor, obs: LocalOp, local_dim=2) -> complex
```

**`Statevector(State)` — constructor:**

```python
Statevector(num_sites: int | None = None, *, bitstring: str | None = None,
            array: np.ndarray | None = None, random_seed: int | None = None,
            local_dim: int = 2)
```

Exactly one of `num_sites`, `bitstring`, `array` must be given. `random_seed` is a modifier of the `num_sites` path (Haar-random state). `bitstring` uses big-endian encoding (`int(bs, local_dim)` → flat index).

**`Statevector` public methods:**

| Method | Signature | Notes |
|---|---|---|
| `expectation` | `(observable) → complex` | Dispatches on `PauliString`, `PauliSum`, `LocalOp`; never forms 2^n matrix |
| `sample` | `(num_shots) → np.ndarray` | `(num_shots, num_sites)` int8 array |
| `measure_and_collapse` | `(sites) → tuple[Statevector, str]` | Project+renorm; returned state keeps all n qubits |
| `overlap` | `(other) → complex` | Routes through `partial_overlap(skip=[])` |
| `rdm` | `(sites) → np.ndarray` | Thin wrapper over `partial_overlap(skip=sites)` |
| `project_and_renorm` | `(sites, outcome_str) → Statevector` | Applies `\|b⟩⟨b\|` projectors, renorms |
| `reset` | `(sites, reset_to=0) → Statevector` | Measure-and-conditional-flip; keeps all n qubits |
| `norm` | `() → float` | Routes through `partial_overlap(skip=[])` |
| `to_array` | `() → np.ndarray` | Flat C-order, shape `(local_dim**num_sites,)` |

**`StatevectorSimulator(Simulator)`:**

```python
class StatevectorSimulator(Simulator):
    def _validate_state(self, state): ...         # raises IncompatibleStateError if not Statevector
    def translate_gate(self, gate): ...           # returns (gate.matrix, gate.indices)
    def _apply_translated_gate(self, state, tg): ...  # state._tensor = op_action(mat, indices, ...)
    # _insert_noise not overridden → inherits IncompatibleNoiseError
```

---

## Design correction: `reset` implementation

The proposal described `reset` as a "thin wrapper over `project_and_renorm`" — this is incorrect.  
`project_and_renorm([0], "0")` on `|1⟩` produces a zero vector (P₀|1⟩ = 0) and cannot be normalized. The implemented `reset` uses **measure-and-conditional-flip**: measure the sites, then apply X at any site whose outcome differs from `reset_to`. This matches the physical reset gate semantics (Qiskit / Cirq convention) and passes `test_statevector_reset`.

---

## Dependency graph (as-built)

```
op_action                           partial_overlap
    │                                     │
    ├── StatevectorSimulator._apply_translated_gate
    │                                     │
    ├── _expectation_pauli_string         ├── Statevector.rdm
    │   (per-site Pauli loop)             │     └── Statevector.measure_and_collapse
    │                                     │
    ├── _expectation_local_op             ├── Statevector.overlap
    │                                     │
    ├── Statevector.project_and_renorm    ├── Statevector.norm
    │     └── Statevector.reset (via m&c) │
    │     └── Statevector.measure_and_collapse
    │                                     └── (final inner product in both
    └── Statevector.reset (X-flip step)        expectation helpers)
```

---

## Tests

42 new tests in `tests/test_backends/test_statevector.py`. Full suite: **201 tests, 0 failures**.

Key physics tests and what they catch:

| Test | What it verifies |
|---|---|
| `test_statevector_sim_ghz` | `op_action` index arithmetic for contiguous 2-qubit gate |
| `test_statevector_sim_ghz_3qubit` | same for 3-qubit chain |
| `test_statevector_expectation_pauli_{z,x,zz,xx}` | site-by-site Pauli contraction; no 2^n matrix |
| `test_statevector_expectation_pauli_coeff` | `PauliString.coeff` applied correctly |
| `test_statevector_rdm_bell` | `partial_overlap(skip=[0])` gives maximally mixed RDM |
| `test_statevector_sample_and_collapse` | Born-rule fractions + correct post-measurement state |
| `test_statevector_measure_and_collapse_bell` | entanglement collapse: measuring one qubit collapses the other |
| `test_statevector_project_and_renorm_two_sites` | multi-site projector |
| `test_statevector_reset_one_to_zero` | measure-and-flip for computational basis state |
| `test_statevector_sim_raises_on_noise` | `IncompatibleNoiseError` propagated from ABC |

**What these tests do NOT catch:**
- Correctness at `local_dim > 2` (qutrit/qudit states): the code supports it but is untested.
- Statistical correctness at high precision (Born-rule tests use 3σ bounds for N=1000/10000).
- Non-Hermitian observables passed to `expectation` (no explicit test).

---

## Checklist for Faisal to verify

- [ ] Open `examples/devbooks/statevector.ipynb` and run cells 2–9. In particular:
  - **Cell 3** (non-contiguous `op_action`): `psi_out[1,0,1]` should print `1.0`
  - **Cell 4** (RDM): `rdm0` should be `[[0.5, 0], [0, 0.5]]`
  - **Cell 7** (`measure_and_collapse`): all 5 trials should print `correct=True`
  - **Cell 9** (dynamic circuit): `Post-correction state is always |00⟩ ✓`
- [ ] Check that `bell.expectation(PauliString("ZZ"))` is `1.0` at the REPL.
- [ ] Check that `Statevector(bitstring="01")._tensor[0,1]` is `1.0`.
