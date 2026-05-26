# Task 7 Proposal — `Statevector` and `StatevectorSimulator`

**Date:** 2026-05-26  
**Status:** Awaiting approval

---

## 1. Overview

Implement `Statevector(State)` and `StatevectorSimulator(Simulator)` in
`backends/statevector.py`. Internal representation is a rank-n NumPy tensor of
shape `[local_dim] * n`. Two public module-level functions — `op_action` and
`partial_overlap` — are the only ncon contraction primitives; every other
operation is a wrapper around one or both of them.

---

## 2. Minor prerequisite changes

### 2.1 `core/observables.py` — add `num_sites`

```python
# PauliString.__init__
self.num_sites = len(self.string)

# PauliSum.__init__
self.num_sites = len(self.terms[0].string)
```

Used in `Statevector.expectation` for the size-mismatch check.

### 2.2 `core/base.py` — rename `sample_and_collapse` → `measure_and_collapse`

```python
@abstractmethod
def measure_and_collapse(self, sites: list[int]) -> tuple[State, str]:
    """Measure sites, collapse state; returns (post-measurement State, outcome_str).
    The returned State has the same number of sites as the original."""
```

Return order flipped to `(State, str)` — state first, matching the
dynamic-circuit loop pattern in the API vision. `sample(num_shots)` is
unchanged. `MinimalState` in `test_base.py` updated to match.

---

## 3. Module-level functions in `backends/statevector.py`

### 3.1 Two primitives (public, ported verbatim)

```python
def op_action(
    op: np.ndarray,
    indices: list[int],
    sv: np.ndarray,
    local_dim: int = 2,
) -> np.ndarray:
    """Apply op (d^k×d^k matrix or rank-2k tensor) at indices to rank-n tensor sv.
    Handles non-contiguous and out-of-order indices via sort+permute before ncon.
    Returns same shape as sv."""
```
→ Port of `op_action`, `legacy/statevector_sim.py` lines 142–158, plus its
helper `locs_to_indices` (lines 128–139) inlined or kept as a private helper.
No semantic change.

```python
def partial_overlap(
    sv1: np.ndarray,
    sv2: np.ndarray,
    local_dim: int = 2,
    skip: list[int] | None = None,
) -> np.ndarray:
    """Partial ⟨sv1|sv2⟩ contracting all sites not in skip.
    skip=None or skip=[] → full overlap, returns (1,1) array.
    Returns (local_dim**|skip|, local_dim**|skip|) matrix."""
```
→ Port of `partial_overlap`, `legacy/statevector_sim.py` lines 178–203.
No semantic change.

Both are public because they are the architectural primitives — every other
operation in this file is a wrapper around one or both of them.

### 3.2 Private helpers

```python
def _locs_to_indices(locs: list[int], n: int) -> tuple[list[int], list[int]]:
    """ncon index labels for a gate at locs on a rank-n state tensor."""
```
→ Port of `locs_to_indices`, `legacy/statevector_sim.py` lines 128–139.
Private because it is purely internal to `op_action`.

### 3.3 Expectation helpers (private, dispatch targets for `Statevector.expectation`)

```python
def _expectation_pauli_string(
    tensor: np.ndarray,
    obs: PauliString,
    local_dim: int = 2,
) -> complex:
    """⟨ψ|P₀⊗P₁⊗…|ψ⟩ without building the full 2^n matrix.

    For each non-I site i: apply _PAULI_MATRICES[obs.string[i]] via op_action.
    Final inner product via partial_overlap(original, right_tensor, skip=[])[0,0].
    """
```
→ Adapted from `pauli_expectation_sv`, `legacy/statevector_sim.py` lines 250–272.
The per-site ncon loop is replaced by `op_action` calls; the final inner product
uses `partial_overlap` rather than an ad-hoc ncon. No 2^n × 2^n matrix formed.

```python
def _expectation_local_op(
    tensor: np.ndarray,
    obs: LocalOp,
    local_dim: int = 2,
) -> complex:
    """⟨ψ|O_sites|ψ⟩ for a LocalOp on obs.indices.

    Applies obs.matrix at obs.indices via op_action, then partial_overlap.
    """
```
→ Generalisation of `one_local_expectation` (lines 92–110) and `local_expectation`
(lines 66–82). Delegates to `op_action` + `partial_overlap`; no embedding.

---

## 4. Dependency graph

```
op_action                        partial_overlap
    │                                  │
    ├── StatevectorSimulator            ├── Statevector.rdm
    │   ._apply_translated_gate        │     └── Statevector.measure_and_collapse
    │                                  │
    ├── _expectation_pauli_string      ├── Statevector.overlap
    │   (per-site loop)                │
    ├── _expectation_local_op          └── (final inner product in both
    │                                       expectation helpers)
    └── Statevector.project_and_renorm
          └── Statevector.reset (wrapper)
              Statevector.measure_and_collapse (collapse step)
```

---

## 5. `Statevector`

### 5.1 Constructor

```python
class Statevector(State):
    def __init__(
        self,
        num_sites: int | None = None,
        *,
        bitstring: str | None = None,
        array: np.ndarray | None = None,
        random_seed: int | None = None,
        local_dim: int = 2,
    ):
```

Exactly one initialisation path active; mixing raises `ValueError`.

| Call | `_tensor` |
|---|---|
| `Statevector(3)` | shape `[2,2,2]`, `_tensor[0,0,0]=1` (all-zero state) |
| `Statevector(bitstring="01")` | `int("01",2)=1` → `_tensor[0,1]=1` |
| `Statevector(array=arr)` | reshaped to rank-n; asserts normalised |
| `Statevector(4, random_seed=42)` | Haar-random via `np.random.default_rng(42)` |

Bitstring encoding: `int(bitstring, local_dim)` index, big-endian (site 0 = MSB),
consistent with `circuits.py`.

### 5.2 Public attributes

- `num_sites: int`
- `local_dim: int`
- `_tensor: np.ndarray` — shape `[local_dim]*num_sites`, dtype `complex128`

### 5.3 ABC methods

```python
@property
def default_simulator(self) -> type[StatevectorSimulator]: ...

def expectation(self, observable: Observable) -> complex:
    """Dispatches on observable type; never builds the full 2^n matrix."""

def sample(self, num_shots: int) -> np.ndarray:
    """Born-rule sampling of full bitstrings; (num_shots, num_sites) int8 array."""

def measure_and_collapse(self, sites: list[int]) -> tuple[Statevector, str]:
    """Sample outcome from rdm diagonal, project and renorm; returns (Statevector, str).
    Returned state has the same num_sites as self."""

def overlap(self, other: Statevector) -> complex:
    """⟨self|other⟩ via partial_overlap(self_flat, other_flat, skip=[])[0,0]."""
```

### 5.4 Extra public methods

```python
def rdm(self, sites: list[int]) -> np.ndarray:
    """Reduced density matrix for sites; (local_dim**|sites|, local_dim**|sites|) matrix.
    Thin wrapper: partial_overlap(self.to_array(), self.to_array(), skip=sites)."""

def project_and_renorm(
    self, sites: list[int], outcome_str: str
) -> Statevector:
    """Apply |bit_i⟩⟨bit_i| projector at each site via op_action, renorm.
    Returns new Statevector with same num_sites. Used by reset and measure_and_collapse."""

def reset(self, sites: list[int], reset_to: int | list[int] = 0) -> Statevector:
    """Thin wrapper over project_and_renorm: builds outcome_str from reset_to, delegates."""

def norm(self) -> float:
    """partial_overlap(flat, flat, skip=[])[0,0].real — should be 1 for a valid state."""

def to_array(self) -> np.ndarray:
    """Flat statevector, shape (local_dim**num_sites,), C-order."""
```

---

## 6. `expectation` dispatch

```python
def expectation(self, observable: Observable) -> complex:
    if isinstance(observable, PauliString):
        if observable.num_sites != self.num_sites:
            raise ValueError(...)
        return _expectation_pauli_string(self._tensor, observable, self.local_dim)
    if isinstance(observable, PauliSum):
        if observable.num_sites != self.num_sites:
            raise ValueError(...)
        return sum(
            _expectation_pauli_string(self._tensor, t, self.local_dim)
            for t in observable.terms
        )
    if isinstance(observable, LocalOp):
        return _expectation_local_op(self._tensor, observable, self.local_dim)
    raise NotImplementedError(f"Statevector.expectation: unsupported {type(observable)}")
```

`Magnetization` is a `PauliSum` subclass — hits the second branch automatically.

---

## 7. `sample`

```python
def sample(self, num_shots: int) -> np.ndarray:
    probs = np.abs(self.to_array()) ** 2
    flat = np.random.default_rng().choice(len(probs), size=num_shots, p=probs)
    bits = (flat[:, None] >> np.arange(self.num_sites - 1, -1, -1)) & 1
    return bits.astype(np.int8)
```

Full-system sampling: amplitude-squared vector is the probability vector directly.
`partial_overlap` is not needed here — no partial trace required.

---

## 8. `measure_and_collapse`

```python
def measure_and_collapse(self, sites: list[int]) -> tuple[Statevector, str]:
    # sampling: reuses rdm-diagonal logic from measure_sv (legacy lines 210-220)
    probs = np.real(np.diag(self.rdm(sorted(sites))))
    outcome_idx = np.random.choice(len(probs), p=probs)
    outcome_str = format(outcome_idx, f'0{len(sites)}b')

    # collapse: project_and_renorm (NOT a port of legacy measure_and_collapse_sv)
    return self.project_and_renorm(sorted(sites), outcome_str), outcome_str
```

The legacy `measure_and_collapse_sv` (lines 223–247) contracts the bra out of
the tensor entirely, returning an (n-k)-qubit state. Here we instead apply
projectors and renorm, keeping all n qubits — required for the dynamic-circuit
re-use pattern.

---

## 9. `StatevectorSimulator`

```python
class StatevectorSimulator(Simulator):

    def _validate_state(self, state: State) -> None:
        if not isinstance(state, Statevector):
            raise IncompatibleStateError(...)

    def translate_gate(self, gate: Gate) -> tuple[np.ndarray, list[int]]:
        """(gate.matrix, gate.indices); reshaping deferred to op_action."""
        return gate.matrix, gate.indices

    def _apply_translated_gate(
        self, state: Statevector, translated_gate: tuple[np.ndarray, list[int]]
    ) -> None:
        mat, indices = translated_gate
        state._tensor = op_action(mat, indices, state._tensor, state.local_dim)
```

`_insert_noise` not overridden — inherits `raise IncompatibleNoiseError`.

---

## 10. Tests (`tests/test_backends/test_statevector.py`)

Required by TODO.md:

| Test | Legacy analogue | What it catches |
|---|---|---|
| `test_statevector_init_bitstring` | `string_to_sv` | `_tensor[0,1]==1` for `"01"` |
| `test_statevector_init_random_seed` | `random_sv` | same tensor from same seed |
| `test_statevector_norm` | `normalize_state` | random SV has norm ≈ 1 |
| `test_statevector_sim_ghz` | `StatevectorSim.run` | H+CNOT on `\|00⟩` → `(1/√2)[1,0,0,1]` |
| `test_statevector_expectation_pauli` | `pauli_expectation_sv` | `⟨Z⟩\|0⟩=1`, `⟨X⟩\|+⟩=1`, `⟨ZZ⟩` Bell=1 |
| `test_statevector_sim_raises_on_noise` | n/a | `IncompatibleNoiseError` from base |
| `test_statevector_sample_and_collapse` | `measure_and_collapse_sv` | Born-rule fractions; collapsed state correct |

Additional:
- `test_statevector_init_allzero` — `Statevector(3)._tensor[0,0,0]==1`
- `test_statevector_overlap_self` — `sv.overlap(sv) ≈ 1`
- `test_statevector_reset` — reset `\|1⟩` to 0 gives `\|0⟩`
- `test_statevector_sample_shape` — `.sample(100)` → `(100, n)` int8
- `test_statevector_expectation_local_op` — `LocalOp(Z,[0])` on `\|0⟩` → 1
- `test_statevector_expectation_magnetization` — `Magnetization(2)` on `\|01⟩` → 0
- `test_statevector_rdm_diagonal` — `rdm([0])` of `\|0⟩` is `[[1,0],[0,0]]`
- `test_statevector_project_and_renorm` — project `\|+⟩` onto "0" gives `\|0⟩`

---

## 11. What will NOT be touched

- `_locs_to_indices` index arithmetic
- `op_action` contraction order (sort+permute+ncon)
- `partial_overlap` label assignment scheme
- Any MPS/MPDO machinery in `_tn_internals.py`
