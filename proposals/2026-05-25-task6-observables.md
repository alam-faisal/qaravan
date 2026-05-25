# Task 6 — Concrete Observable subclasses in `core/observables.py`

## Approach

Mirror the `Gate` pattern: the `Observable` ABC gets an abstract `matrix` property, and every concrete subclass implements it. Backends build their own preferred representations from `obs.matrix`, `obs.indices`, and `obs.as_pauli_sum()` — observables don't handle that themselves.

One intentional asymmetry: `as_pauli_sum()` remains a regular method (not a property), because only some observables support it and its signature may evolve.

`Magnetization` is a subclass of `PauliSum` — it constructs its own terms and inherits `.matrix` and `.as_pauli_sum()` for free.

One small change to `core/base.py`: add `matrix` as an abstract property to `Observable`.

---

## Changes to `core/base.py`

```python
class Observable(ABC):
    def __init__(self, name: str, indices: int | list[int]):
        self.name = name
        self.indices = [indices] if isinstance(indices, int) else list(indices)

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Local or full Hermitian matrix for this observable."""
        ...

    def __str__(self) -> str:
        return f"{self.name}({self.indices})"
```

---

## Classes and signatures

### `PauliString`

```python
class PauliString(Observable):
    def __init__(self, string: str, coeff: complex = 1.0)
```

- `string`: Pauli letters, e.g. `"XZI"` (case-insensitive, normalised to upper internally). Length = num sites; `I` = identity.
- `indices`: `[0, 1, ..., len(string)-1]`.
- `coeff`: scalar weight.
- `matrix` property: `coeff * kron(P_0, P_1, ..., P_{n-1})`, shape `(2^n, 2^n)`.
- `as_pauli_sum() -> PauliSum`: wraps self in a length-1 PauliSum.
- Arithmetic: `__mul__`/`__rmul__` by scalar → new PauliString with scaled coeff. `__add__` with PauliString or PauliSum → PauliSum.
- `__repr__`: `"XZI"` if coeff=1, else `"(0.5+0j)*XZI"`.

### `PauliSum`

```python
class PauliSum(Observable):
    def __init__(self, terms: list[PauliString])
```

- `terms`: all must have the same string length. Raises `ValueError` otherwise.
- `indices`: `[0, ..., n-1]` where n = string length.
- `matrix` property: `sum(t.matrix for t in terms)`, shape `(2^n, 2^n)`.
- `as_pauli_sum() -> PauliSum`: returns `self`.
- `__add__` with PauliString or PauliSum → new PauliSum (no mutation).
- `__mul__`/`__rmul__` by scalar → new PauliSum with all coeffs scaled.

### `LocalOp`

```python
class LocalOp(Observable):
    def __init__(self, matrix: np.ndarray, indices: int | list[int])
```

- `matrix`: `d^k × d^k` Hermitian operator on the sites listed in `indices`. No embedding.
- `matrix` property: returns the stored matrix as-is.
- The backend is responsible for embedding into the full Hilbert space using `obs.indices`.

### `Magnetization(PauliSum)`

```python
class Magnetization(PauliSum):
    def __init__(self, n: int, axis: str = "Z")
```

- Builds terms `[(1/n) * PauliString("I"*i + axis + "I"*(n-i-1)) for i in range(n)]` and calls `super().__init__(terms)`.
- `n` and `axis` stored as attributes.
- `.matrix` and `.as_pauli_sum()` both inherited from `PauliSum` — no override needed.

---

## Tests in `tests/test_core/test_observables.py`

```
test_pauli_string_as_matrix
    PauliString("I").matrix == I_2
    PauliString("X").matrix == X_2
    PauliString("Y").matrix == Y_2
    PauliString("Z").matrix == Z_2
    PauliString("XZ").matrix == kron(X, Z)
    PauliString("XZ", coeff=2.0).matrix == 2*kron(X, Z)
    lowercase "xz" also accepted

test_pauli_sum_as_matrix
    PauliSum([PauliString("IZ"), PauliString("ZI")]).matrix == kron(I,Z) + kron(Z,I)

test_pauli_string_arithmetic
    2.0 * PauliString("X") has coeff=2 and correct matrix
    PauliString("X") + PauliString("Z") is PauliSum with two terms

test_local_op_matrix
    LocalOp(Z_mat, [0]).matrix == Z_mat  (no embedding)
    LocalOp(Z_mat, [1]).matrix == Z_mat  (still just the local matrix)
    obs.indices == [1]

test_magnetization_on_basis_states
    # Direct matrix multiply, no simulator needed
    M = Magnetization(2)
    |00⟩: np.array([1,0,0,0]) @ M.matrix @ np.array([1,0,0,0]) == 1.0
    |11⟩: == -1.0
    |01⟩: == 0.0
    isinstance(Magnetization(2), PauliSum) is True
```

---

## Remaining open question

**`PauliSum` naming conflict**: `backends/pauli_propagation.py` will have a `PauliSum` State. Flag this when that task is reached; likely rename the state to `PauliDict` or `PauliSumState`.
