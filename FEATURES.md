# Qaravan — Implemented Features (v0.2.1)

This document lists what is currently implemented in `src/qaravan/` (excluding `legacy/`). Stubs are noted explicitly.

---

## Core abstractions — `core/base.py`

### `Gate` (ABC)
- `name`, `indices` (int auto-normalised to `list[int]`), `time`
- Abstract: `matrix` (d^k × d^k), `dagger()`
- Derived: `num_sites`, `local_dim`

### `State` (ABC)
- Abstract: `default_simulator`, `expectation(observable)`, `sample(num_shots)`, `measure_and_collapse(sites)`, `overlap(other)`, `__repr__`
- Concrete: `__str__`, `apply(circuit)` — evolves via `default_simulator`

### `Simulator` (ABC)
- `__init__(circuit, initial_state, noise_model=None, decompose=False)` — validates state at construction
- `run()` pipeline: deep-copy state → copy circuit → optional `decompose()` → `construct_layers()` → `_insert_noise()` → translate+apply each gate → return state
- Abstract: `_validate_state`, `translate_gate`, `_apply_translated_gate`
- Default `_insert_noise`: raises `IncompatibleNoiseError` if noise model is not None

### `Observable` (ABC)
- `name`, `indices` (int auto-normalised)
- Abstract: `matrix`

### `NoiseModel` (ABC)
- Abstract: `gate_dependent` (property)
- `get_kraus()` — raises `NotImplementedError` by default; subclasses override
- `get_superop()` — implemented in terms of `get_kraus`: Σ_k (K_k* ⊗ K_k)
- `sample_error()` — raises `NotImplementedError` by default

### Exceptions
- `IncompatibleNoiseError` — raised when a noise-unaware backend receives a noise model
- `IncompatibleStateError` (subclass of `TypeError`) — raised when a simulator receives the wrong state type

---

## Gates — `core/gates.py`

### Base classes
| Class | Description |
|---|---|
| `MatrixGate` | Fixed matrix; use for ad-hoc gates. `dagger()` conjugate-transposes. |
| `ParametricGate` | Abstract; subclasses implement `_build_matrix()`. `dagger()` negates all params. Convention: exp(−iθP). |

### Single-qubit fixed gates
`I`, `X`, `Y`, `Z`, `H`, `S`, `Sdg`, `T`, `Tdg`, `SX`

`S`/`Sdg` and `T`/`Tdg` know their inverses: `S(0).dagger()` returns `Sdg`, not a generic `MatrixGate`.

### Single-qubit parametric gates
Convention: exp(−iθP), full angle (no hidden ½ factor).

| Gate | Matrix |
|---|---|
| `RX(i, θ)` | cos(θ)I − i sin(θ)X |
| `RY(i, θ)` | cos(θ)I − i sin(θ)Y |
| `RZ(i, θ)` | diag(e^{−iθ}, e^{iθ}) |

### Two-qubit fixed gates
`CNOT`, `CZ`, `SWAP`, `iSWAP` — all in (control, target) / (site0, site1) big-endian convention.

### Two-qubit parametric gates
Convention: exp(−iθP⊗P), full angle.

| Gate | Matrix |
|---|---|
| `RXX([i,j], θ)` | exp(−iθ X⊗X) |
| `RYY([i,j], θ)` | exp(−iθ Y⊗Y) |
| `RZZ([i,j], θ)` | exp(−iθ Z⊗Z) = diag(e^{−iθ}, e^{iθ}, e^{iθ}, e^{−iθ}) |

### Qutrit gates (local_dim = 3, inferred from matrix shape)
Single-qutrit: `X01`, `SX01`, `X12`, `SX12`, `H01`, `SDG01`

Two-qutrit: `SWAP3`, `CNOT3` (control, target convention)

### Utilities
| Function | Description |
|---|---|
| `random_unitary(n, local_dim=2, seed=None)` | Haar-random d^n × d^n unitary |
| `is_unitary(u, atol=1e-10)` | Bool unitarity check |
| `pauli_string_to_gates(string)` | `"XZI"` → `[X(0), Z(1)]`; case-insensitive, identity sites skipped |

---

## Circuit — `core/circuits.py`

### `Circuit(gates, num_sites=None, local_dim=2)`
`num_sites` inferred from gate indices if not given.

| Method / property | Description |
|---|---|
| `construct_layers()` | Greedy parallel packing: each gate goes to the earliest layer with no qubit conflict |
| `dagger()` | Reversed gate order, each gate conjugate-transposed |
| `to_matrix()` | Full unitary matrix (debug only; real backends never call this) |
| `num_params` | Total continuous parameter count across all `ParametricGate`s |
| `bind(params)` | Return new `Circuit` with ParametricGate params replaced sequentially |
| `copy()` | Shallow gate-list copy; layers reset to None |
| `+` | Concatenate two circuits |
| `*`, `__rmul__` | Repeat circuit n times |
| `[i]` | Returns `Gate` |
| `[i:j]` | Returns `Circuit` |
| `decompose()` | **Not yet implemented** |

### `_embed_gate(gate, num_sites, local_dim)`
Lifts a k-site gate matrix into the full num_sites-site Hilbert space. Handles both contiguous (kron with identity padding) and non-contiguous / out-of-order site indices (permutation-based approach). Used by `to_matrix()` and debug code.

---

## Observables — `core/observables.py`

### `PAULI_MATRICES`
Public dict `{"I", "X", "Y", "Z"}` → complex128 2×2 arrays.

### `PauliString(string, coeff=1.0)`
Weighted tensor product of single-qubit Paulis (e.g. `PauliString("XZI", coeff=0.5)`).
- `matrix` property: full 2^n × 2^n matrix via `np.kron`
- Arithmetic: `p1 + p2` → `PauliSum`, `scalar * p` → rescaled `PauliString`
- `as_pauli_sum()` → `PauliSum`

### `PauliSum(terms)`
Linear combination of `PauliString`s; all terms must have the same length.
- `matrix` property: sum of term matrices
- Arithmetic: `+` (PauliString or PauliSum), `*` scalar

### `LocalOp(matrix, indices)`
Arbitrary dense Hermitian operator on specified sites. Matrix stored as-is; backends use `obs.indices` for placement.

### `Magnetization(n, axis='Z')`
(1/n) Σᵢ σᵢ^axis. Subclass of `PauliSum`. Axis can be X, Y, or Z.

---

## Statevector backend — `backends/statevector.py`

### ncon primitives
| Function | Description |
|---|---|
| `op_action(op, indices, sv, local_dim=2)` | Apply d^k × d^k operator at arbitrary (non-contiguous, out-of-order) site indices to a rank-n tensor or flat vector. Returns same shape as input. |
| `partial_overlap(sv1, sv2, local_dim=2, skip=None)` | Partial ⟨sv1\|sv2⟩ contracting all sites not in `skip`. Returns (d^\|skip\| × d^\|skip\|) matrix; `skip=None` gives (1,1) full overlap. |

### `Statevector(State)`
Internal representation: rank-n complex128 tensor `_tensor`.

**Construction** — exactly one of:
```python
Statevector(num_sites)               # |0…0⟩ or random if random_seed given
Statevector(bitstring="01")          # big-endian; int("01", local_dim) index
Statevector(array=arr)               # from flat array; norm-checked
```
All paths accept `local_dim` (default 2).

**Methods:**
| Method | Description |
|---|---|
| `expectation(obs)` | PauliString, PauliSum, LocalOp — never builds 2^n matrix; raises `NotImplementedError` for unknown types |
| `sample(num_shots)` | Born-rule sampling; returns `(num_shots, num_sites)` int8 array |
| `measure_and_collapse(sites)` | Sample outcome from RDM diagonal, project+renorm; returns `(Statevector, outcome_str)`; state retains original `num_sites` |
| `overlap(other)` | ⟨self\|other⟩ via `partial_overlap` |
| `rdm(sites)` | Reduced density matrix; returns (d^\|sites\| × d^\|sites\|) array |
| `project_and_renorm(sites, outcome_str)` | Apply \|bit⟩⟨bit\| projectors, renormalize |
| `reset(sites, reset_to=0)` | Measure sites, conditionally flip to target via X gate; `reset_to` can be int or list |
| `norm()` | Should be 1.0 for a valid state |
| `to_array()` | Flat statevector, shape (local_dim^num_sites,), C-order |
| `__repr__` | Bra-ket notation: `0.7071\|00⟩ + 0.7071\|11⟩`; works for any local_dim |

### `StatevectorSimulator(Simulator)`
- Validates initial state is `Statevector` (raises `IncompatibleStateError` otherwise)
- `translate_gate`: returns `(matrix, indices)` tuple
- `_apply_translated_gate`: in-place `op_action` on `state._tensor`
- Noise: inherits `IncompatibleNoiseError` — no noise support at this backend

---

## Circuit library — `applications/circuit_library.py`

| Function | Description |
|---|---|
| `nn_pairs(n, periodic=False)` | Nearest-neighbor index pairs for a 1D chain |
| `ghz_circuit(n)` | H(0) + CNOT(0,1) + … + CNOT(n-2, n-1) |
| `rx_layer(n, params=None, seed=None)` | n-site RX layer; random params if not given |
| `ry_layer(n, params=None, seed=None)` | n-site RY layer |
| `rz_layer(n, params=None, seed=None)` | n-site RZ layer |
| `rxx_layer(skeleton, params=None, seed=None)` | RXX gates on each pair in skeleton |
| `ryy_layer(skeleton, params=None, seed=None)` | RYY gates on each pair in skeleton |
| `rzz_layer(skeleton, params=None, seed=None)` | RZZ gates on each pair in skeleton |

---

## Not yet implemented (stubs)

| Module | Content |
|---|---|
| `core/noise.py` | `ThermalNoise`, `PauliNoise`, `PauliLindbladNoise` (Task 8) |
| `core/hamiltonians.py` | `TFI`, `XXZ`, `Heisenberg` (Task 10) |
| `core/lattices.py` | `Linear`, `Square`, toric, Kagome lattice helpers |
| `backends/density_matrix.py` | `DensityMatrix`, `DensityMatrixSimulator` (Task 9) |
| `backends/mps.py` | `MPS`, `MPSSimulator` |
| `backends/mpdo.py` | `MPDO`, `MPDOSimulator` |
| `backends/monte_carlo.py` | `Walkers`, `MonteCarloSimulator` |
| `backends/clifford.py` | `StabilizerTableau`, `CliffordSimulator` (wraps Stim) |
| `backends/matchgate.py` | `GaussianState`, `MatchgateSimulator` |
| `backends/pauli_propagation.py` | `PauliPropagationSimulator` (wraps PauliPropagation.jl) |
| `Circuit.decompose()` | Gate decomposition into native basis |
