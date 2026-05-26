"""Concrete Observable subclasses: PauliString, PauliSum, LocalOp, Magnetization."""

from __future__ import annotations
import numpy as np
from functools import reduce
from qaravan.core.base import Observable

_PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


class PauliString(Observable):
    """Weighted tensor product of single-qubit Paulis, e.g. PauliString("XZI", coeff=0.5).

    Indices are [0, ..., len(string)-1]. Accepts upper or lower case.
    """

    def __init__(self, string: str, coeff: complex = 1.0):
        self.string = string.upper()
        if any(c not in _PAULI_MATRICES for c in self.string):
            raise ValueError(f"Invalid Pauli characters in '{string}'; use I, X, Y, Z.")
        self.coeff = complex(coeff)
        n = len(self.string)
        self.num_sites = n
        super().__init__(f"P({self.string})", list(range(n)))

    @property
    def matrix(self) -> np.ndarray:
        mats = [_PAULI_MATRICES[c] for c in self.string]
        return self.coeff * reduce(np.kron, mats)

    def as_pauli_sum(self) -> PauliSum:
        return PauliSum([self])

    def __mul__(self, scalar: complex) -> PauliString:
        return PauliString(self.string, self.coeff * scalar)

    def __rmul__(self, scalar: complex) -> PauliString:
        return self.__mul__(scalar)

    def __add__(self, other: PauliString | PauliSum) -> PauliSum:
        if isinstance(other, PauliString):
            return PauliSum([self, other])
        if isinstance(other, PauliSum):
            return PauliSum([self, *other.terms])
        return NotImplemented

    def __radd__(self, other: PauliString | PauliSum) -> PauliSum:
        if isinstance(other, PauliString):
            return PauliSum([other, self])
        if isinstance(other, PauliSum):
            return PauliSum([*other.terms, self])
        return NotImplemented

    def __repr__(self) -> str:
        if self.coeff == 1.0:
            return self.string
        return f"({self.coeff})*{self.string}"


class PauliSum(Observable):
    """Linear combination of PauliStrings. All terms must have the same length."""

    def __init__(self, terms: list[PauliString] | list[str]):
        if not terms:
            raise ValueError("PauliSum requires at least one term.")
        terms = [t if isinstance(t, PauliString) else PauliString(t) for t in terms]
        lengths = {len(t.string) for t in terms}
        if len(lengths) > 1:
            raise ValueError(
                f"All PauliString terms must have the same length; got {lengths}."
            )
        n = len(terms[0].string)
        self.num_sites = n
        self.terms = list(terms)
        super().__init__("PauliSum", list(range(n)))

    @property
    def matrix(self) -> np.ndarray:
        return sum(t.matrix for t in self.terms)  # type: ignore[return-value]

    def as_pauli_sum(self) -> PauliSum:
        return self

    def __add__(self, other: PauliString | PauliSum) -> PauliSum:
        if isinstance(other, PauliString):
            return PauliSum([*self.terms, other])
        if isinstance(other, PauliSum):
            return PauliSum([*self.terms, *other.terms])
        return NotImplemented

    def __radd__(self, other: PauliString | PauliSum) -> PauliSum:
        if isinstance(other, PauliString):
            return PauliSum([other, *self.terms])
        if isinstance(other, PauliSum):
            return PauliSum([*other.terms, *self.terms])
        return NotImplemented

    def __mul__(self, scalar: complex) -> PauliSum:
        return PauliSum([scalar * t for t in self.terms])

    def __rmul__(self, scalar: complex) -> PauliSum:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return " + ".join(repr(t) for t in self.terms)


class LocalOp(Observable):
    """Generic local Hermitian operator on specified sites.

    matrix is returned as-is (no embedding). Backends use obs.indices to
    embed it into the full Hilbert space.
    """

    def __init__(self, mat: np.ndarray, indices: int | list[int]):
        self._matrix = np.asarray(mat, dtype=complex)
        super().__init__("LocalOp", indices)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix


class Magnetization(PauliSum):
    """(1/n) * sum_{i=0}^{n-1} sigma_i^axis. Subclass of PauliSum.

    axis: 'X', 'Y', or 'Z' (case-insensitive).
    """

    def __init__(self, n: int, axis: str = "Z"):
        axis = axis.upper()
        if axis not in ("X", "Y", "Z"):
            raise ValueError(f"axis must be X, Y, or Z; got '{axis}'.")
        self.n = n
        self.axis = axis
        terms = [
            PauliString("I" * i + axis + "I" * (n - i - 1), coeff=1.0 / n)
            for i in range(n)
        ]
        super().__init__(terms)
        self.name = f"Magnetization{axis}"
