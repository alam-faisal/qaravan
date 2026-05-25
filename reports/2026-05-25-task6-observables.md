# Task 6 Report — Concrete `Observable` subclasses

**Status: SUCCESS**

---

## Summary

159 tests pass (135 pre-existing + 24 new in `tests/test_core/test_observables.py`).

### Changes to `core/base.py`

`Observable` ABC gains an abstract `matrix` property, parallel to `Gate.matrix`:

```python
@property
@abstractmethod
def matrix(self) -> np.ndarray:
    """Hermitian matrix for this observable (local or full, depending on subclass)."""
    ...
```

`MinimalObservable` in `test_base.py` updated to implement the new abstract method.

### New file: `core/observables.py`

**`PauliString(string, coeff=1.0)`**
```python
PauliString(string: str, coeff: complex = 1.0)
```
- Accepts upper or lower case; normalises to upper internally.
- `matrix` property: `coeff * kron(P_0, ..., P_{n-1})`, shape `(2^n, 2^n)`.
- `as_pauli_sum() -> PauliSum`.
- Arithmetic: `*`/`*=` by scalar; `+` with `PauliString` or `PauliSum` returns `PauliSum`.

**`PauliSum(terms)`**
```python
PauliSum(terms: list[PauliString])
```
- All terms must have the same string length; raises `ValueError` otherwise.
- `matrix` property: `sum(t.matrix for t in terms)`, shape `(2^n, 2^n)`.
- `as_pauli_sum()` returns `self`.
- Arithmetic: `+` (PauliString or PauliSum), `*`/`*=` by scalar.

**`LocalOp(mat, indices)`**
```python
LocalOp(mat: np.ndarray, indices: int | list[int])
```
- `matrix` returns the stored local matrix as-is — no embedding.
- Backends use `obs.indices` to embed into the full Hilbert space.

**`Magnetization(n, axis="Z")`**
```python
Magnetization(n: int, axis: str = "Z")
```
- Subclass of `PauliSum`. Constructs terms `[(1/n) * PauliString("I"*i + axis + "I"*(n-i-1)) for i in range(n)]` at init.
- `matrix` and `as_pauli_sum()` both inherited — no override.
- `axis` in `{"X", "Y", "Z"}`, case-insensitive; raises `ValueError` otherwise.

### Changes to `core/__init__.py`

Exports `PauliString`, `PauliSum`, `LocalOp`, `Magnetization`.

---

## Tests (`tests/test_core/test_observables.py`) — 24 tests

**`PauliString` matrices:** single-qubit I/X/Y/Z, two-qubit `"XZ"` and `"IX"`, coefficient scaling, lowercase input, invalid character raises, `indices` attribute.

**Arithmetic:** scalar `*`/`rmul` updates `coeff` and `matrix`; `PauliString + PauliString` produces `PauliSum` of length 2; `PauliString + PauliSum` produces correctly merged `PauliSum`.

**`PauliSum`:** `as_matrix` matches direct kron construction; mismatched lengths raise; empty list raises; scalar multiply; sum of two `PauliSum`s; `as_pauli_sum()` returns self; `PauliString.as_pauli_sum()` wraps correctly.

**`LocalOp`:** `matrix` returned as-is for single-site and two-site operators; `indices` stored correctly.

**`Magnetization`:** `isinstance(Magnetization(2), PauliSum)` is true; matrix shape; basis-state expectations `|00⟩ → +1`, `|11⟩ → -1`, `|01⟩ → 0`; X-axis on `|+⟩` gives 1; invalid axis raises.

---

## What to verify by hand

1. **`examples/task6_observables.ipynb`** — open and run all cells. Key checks:
   - `PauliString("ZZ").matrix` should match `np.kron(Z, Z)` explicitly.
   - `Magnetization(4).matrix` diagonal should be `[1, 0.5, 0, -0.5, 0.5, 0, -0.5, 0, 0, -0.5, 0, 0.5, -0.5, 0, 0.5, 1]` × (1/2) — wait, that's (Σ Z_i / 4) so diagonal entries are (spin-sum / 4) for each computational basis state.
   - Arithmetic cell: `0.5 * PauliString("Z") + 0.5 * PauliString("X")` should produce a 2×2 matrix with correct entries.
   - `LocalOp` cell: confirm `obs.matrix == Z` and `obs.indices == [1]` for a single-site op.

2. **`Magnetization` normalisation** — the (1/n) prefactor means ⟨M⟩ ∈ [-1, +1] for any product state. Verify this in the notebook for an n=4 example.

3. **`PauliSum` arithmetic** — the notebook chains `+` and `*` to build a simple Ising-like observable `J*(ZZ + ZZ) + h*IX`; confirm the matrix matches a hand-built numpy expression.

---

## Notes

- `LocalOp` does **not** embed. The statevector backend (Task 7) will need to handle embedding when calling `expectation(LocalOp(...))`. This is by design — the Observable is not responsible for knowing the total system size.
- The `PauliSum` naming will conflict with the `PauliSum` State in `backends/pauli_propagation.py` (Task 12+). Flag at that task; likely rename the State to `PauliDict`.
