# Proposal: Task 5 — Circuit generation functions in `core/circuits.py`

## Approach

The `Circuit` class itself is done. This task adds **free functions** that generate common circuit instances, porting the relevant parts of `legacy/circuits.py` and adding a few new ones. All functions return a `Circuit`; none mutate existing ones.

Scope is deliberately narrow: named structural circuits (`ghz_circuit`), rotation-layer factories, a nearest-neighbor skeleton helper, and nothing that depends on objects not yet in v0.2 (no `kak_unitary`, no `HeisenbergCoupling`, no brickwall variational ansatze — those belong in `applications/` once those gates exist).

---

## Functions

### `nn_pairs(n: int, periodic: bool = False) -> list[list[int]]`

Returns `[[0,1], [1,2], ..., [n-2, n-1]]`, plus `[n-1, 0]` if `periodic=True`. Used to define 1D nearest-neighbor skeletons for layer generators and Trotter circuits. Small but eliminates a recurring boilerplate pattern.

### `ghz_circuit(n: int) -> Circuit`

`H(0)` followed by `CNOT(i, i+1)` for `i` in `0..n-2`. The canonical $n$-qubit entangled state prep circuit. Useful as a reference circuit throughout the codebase.

### Single-site rotation layers

```python
def rx_layer(n: int, params: np.ndarray | None = None, seed: int | None = None) -> Circuit
def ry_layer(n: int, params: np.ndarray | None = None, seed: int | None = None) -> Circuit
def rz_layer(n: int, params: np.ndarray | None = None, seed: int | None = None) -> Circuit
```

`params=None` draws uniformly from `[0, 2π)` via `np.random.default_rng(seed)`. If `params` is provided, `seed` is ignored. The legacy used `np.random.rand` without a seed argument; adding `seed` is the only change.

### Two-site rotation layers

```python
def rxx_layer(skeleton: list[list[int]], params: np.ndarray | None = None, seed: int | None = None) -> Circuit
def ryy_layer(skeleton: list[list[int]], params: np.ndarray | None = None, seed: int | None = None) -> Circuit
def rzz_layer(skeleton: list[list[int]], params: np.ndarray | None = None, seed: int | None = None) -> Circuit
```

`skeleton` is a list of index pairs, e.g. `nn_pairs(n)`. Same `params`/`seed` logic as single-site layers.

---

## Tests (`tests/test_core/test_circuits.py`)

### Structure tests

- `test_nn_pairs_open`: `nn_pairs(4) == [[0,1],[1,2],[2,3]]`
- `test_nn_pairs_periodic`: `nn_pairs(4, periodic=True)[-1] == [3, 0]`
- `test_nn_pairs_two_sites`: `nn_pairs(2) == [[0, 1]]`
- `test_ghz_circuit_structure`: first gate is H on index 0; remaining n-1 gates are CNOTs; `len == n`
- `test_ghz_circuit_num_sites`: `ghz_circuit(4).num_sites == 4`
- `test_rx_layer_num_gates`: `len(rx_layer(5)) == 5`
- `test_rx_layer_indices`: gate `i` acts on site `i`
- `test_rx_layer_is_rx`: each gate is an instance of `RX`
- `test_rz_layer_provided_params`: gate `i` has `params[i]` as its rotation angle
- `test_rz_layer_seed_reproducible`: same seed gives identical circuit
- `test_rzz_layer_num_gates`: matches skeleton length
- `test_rzz_layer_skeleton_indices`: gate `i` has indices matching `skeleton[i]`
- `test_rzz_layer_provided_params`: gate `i` has `params[i]` as its rotation angle

### Physics tests (using `Circuit.to_matrix()`)

- `test_ghz_n2_bell_state`: `ghz_circuit(2).to_matrix() @ |00⟩ ≈ (|00⟩ + |11⟩)/√2`
- `test_ghz_n3_state`: `ghz_circuit(3).to_matrix() @ |000⟩ ≈ (|000⟩ + |111⟩)/√2`
- `test_rx_layer_at_zero_is_identity`: `rx_layer(2, params=[0, 0]).to_matrix() ≈ I₄`
- `test_rzz_layer_diagonal`: `rzz_layer([[0,1]], params=[θ])` matrix diagonal matches `exp(-iθ·ZZ/2)` expected values — cross-checks the RZZ convention from gates.py

---

## `Circuit.bind` and `Circuit.num_params`

### `ParametricGate.num_params` attribute

Set in `ParametricGate.__init__` as `self.num_params = len(self.params)`. Plain attribute — `params` is always provided at construction time.

### `Circuit.num_params` property

```python
@property
def num_params(self) -> int:
    return sum(g.num_params for g in self.gates if isinstance(g, ParametricGate))
```

Requires importing `ParametricGate` from `gates.py` in `circuits.py`. No circular import: `circuits.py → gates.py → base.py`; `base.py`'s `Circuit` import is TYPE_CHECKING-only.

### `Circuit.bind(params) -> Circuit`

```python
def bind(self, params: np.ndarray | list[float]) -> Circuit:
    assert len(params) == self.num_params
    new_gates, idx = [], 0
    for gate in self.gates:
        if isinstance(gate, ParametricGate):
            new_gates.append(type(gate)(gate.indices, *params[idx : idx + gate.num_params], gate.time))
            idx += gate.num_params
        else:
            new_gates.append(gate)
    return Circuit(new_gates, num_sites=self.num_sites, local_dim=self.local_dim)
```

Sequential assignment: parameters are consumed gate by gate in circuit order. Works for any `num_params` per gate as long as each gate's constructor accepts `(indices, *params, time)`, which is the convention for all current and planned parametric gates.

### New tests for `bind` / `num_params`

- `test_num_params_all_parametric`: circuit of 3 RX gates → `num_params == 3`
- `test_num_params_mixed`: circuit of RX + H + RZZ → `num_params == 2`
- `test_num_params_no_parametric`: circuit of H, CNOT → `num_params == 0`
- `test_bind_updates_params`: `bind([θ₁, θ₂])` gives gates with those angles
- `test_bind_wrong_count_raises`: `bind` with wrong length raises `AssertionError`
- `test_bind_original_unchanged`: original circuit's params unmodified after bind
- `test_bind_non_parametric_preserved`: MatrixGate in circuit appears unchanged in bound circuit
- `test_bind_physics`: `rx_layer(1, params=[0]).bind([π/2]).to_matrix() ≈ RX(0, π/2).matrix`

---

## Edge cases

- `nn_pairs(1)` returns `[]` (no pairs for a single site); `ghz_circuit(1)` returns a 1-gate circuit with just `H(0)`.
- Empty params array passed to a layer generator should return a zero-gate circuit. Actually, `rx_layer(0)` returns an empty circuit — `params` would be length 0 too.
- `rzz_layer([], params=[])` → empty circuit, no error.

---

## What I'm not sure about

Nothing major. One minor question: should `rx_layer` etc. live at module level in `circuits.py`, or in a submodule? I'd put them directly in `circuits.py` — there aren't many and they're tightly coupled to the `Circuit` constructor. Let me know if you'd prefer a separate `core/circuit_library.py`.
