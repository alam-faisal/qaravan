# Task 5 Report — Circuit generation functions + `bind`/`num_params`

**Status: SUCCESS**

---

## Summary

165 tests pass (125 pre-existing + 40 new in `tests/test_core/test_circuits.py`).

### Changes to `core/circuits.py`

**New import:** `ParametricGate`, `CNOT`, `H`, `RX`, `RY`, `RZ`, `RXX`, `RYY`, `RZZ` from `gates.py`. No circular import — `gates.py → base.py`; `base.py`'s `Circuit` reference is `TYPE_CHECKING`-only.

**`Circuit.num_params` property**
```python
@property
def num_params(self) -> int:
    return sum(g.num_params for g in self.gates if isinstance(g, ParametricGate))
```

**`Circuit.bind(params) -> Circuit`**
```python
def bind(self, params: np.ndarray | list[float]) -> Circuit
```
Sequential assignment: parameters consumed gate by gate in circuit order. `AssertionError` if `len(params) != self.num_params`. Returns a new `Circuit`; original is unchanged. Works for any `num_params` per gate as long as the gate's constructor accepts `(indices, *params, time)`.

**New circuit generation functions:**

| Function | Signature | Notes |
|---|---|---|
| `nn_pairs` | `(n, periodic=False) -> list[list[int]]` | 1D NN skeleton helper |
| `ghz_circuit` | `(n) -> Circuit` | H(0) + chain of CNOTs |
| `rx_layer` | `(n, params=None, seed=None) -> Circuit` | n RX gates, site i→i |
| `ry_layer` | `(n, params=None, seed=None) -> Circuit` | n RY gates |
| `rz_layer` | `(n, params=None, seed=None) -> Circuit` | n RZ gates |
| `rxx_layer` | `(skeleton, params=None, seed=None) -> Circuit` | RXX on skeleton pairs |
| `ryy_layer` | `(skeleton, params=None, seed=None) -> Circuit` | RYY on skeleton pairs |
| `rzz_layer` | `(skeleton, params=None, seed=None) -> Circuit` | RZZ on skeleton pairs |

All layer generators: `params=None` draws uniform `[0, 2π)` via `np.random.default_rng(seed)`.

### Changes to `core/gates.py`

`ParametricGate.__init__` now sets `self.num_params = len(self.params)` — plain attribute required by `Circuit.bind`.

---

## Tests (`tests/test_core/test_circuits.py`) — 40 tests

**`nn_pairs`:** open chain, periodic, n=1 (empty), n=2 periodic.

**`ghz_circuit` structure:** length, first gate is H(0), remaining are CNOTs with correct indices, `num_sites`.

**`ghz_circuit` physics:** `ghz_circuit(2).to_matrix() @ |00⟩ ≈ (|00⟩+|11⟩)/√2` and n=3 analogue. These are the only tests that combine circuit generation with matrix action.

**Layer generators — structure:** gate count, site indices, gate types, provided params stored correctly, seed reproducibility, empty input.

**Layer generators — physics:** `rx_layer(2, params=[0,0]).to_matrix() ≈ I₄`; `rzz_layer([[0,1]], [θ])` diagonal matches `exp(-iθ·ZZ)` convention from Task 4.

**`num_params`:** all-parametric, mixed, none, empty circuit.

**`bind`:** params updated, wrong count raises, original unchanged, non-parametric gates preserved, return type is `Circuit`, physics check `rx_layer(1,[0]).bind([π/2]).to_matrix() ≈ RX(0,π/2).matrix`.

---

## What to verify by hand

1. **`examples/task5_circuits.ipynb`** — open and run all cells. Key checks:
   - GHZ Bell state probabilities: after `ghz_circuit(3).to_matrix() @ |000⟩`, only `|000⟩` and `|111⟩` amplitudes should be nonzero at `1/√2`.
   - `bind` workflow cell: the circuit printed before and after bind should show different parameter values; `num_params` should be the same before and after.
   - `rzz_layer` diagonal cell: manually compare phases against `exp(-iθ)` / `exp(+iθ)` pattern.

2. **`bind` convention** — parameters are assigned left-to-right in circuit gate order, not sorted by site index. The notebook shows a mixed circuit (RX, H, RZZ) to make this explicit.

3. **Random layer reproducibility** — `rz_layer(4, seed=42)` called twice should give identical parameter values; the notebook verifies this.

---

## Notes

- `ghz_circuit(1)` returns a single H(0) gate — technically correct but worth knowing if you use it in a loop over n.
- `bind` passes `gate.time` through to the new gate, preserving any timing metadata on the original parametric gate.
- `rzz_layer` with `nn_pairs(n)` is the building block for 1D ZZ Trotter steps; the combination is demonstrated in the notebook.
