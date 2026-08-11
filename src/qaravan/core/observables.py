"""Concrete Observable subclasses: PauliString, PauliSum, LocalOp, Magnetization."""

from __future__ import annotations
import numpy as np
from functools import reduce
from qaravan.core.base import Observable

PAULI_MATRICES: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

_PAULI_MULT: dict[tuple[str, str], tuple[complex, str]] = {
    ("I", "I"): (1, "I"),
    ("I", "X"): (1, "X"),
    ("I", "Y"): (1, "Y"),
    ("I", "Z"): (1, "Z"),
    ("X", "I"): (1, "X"),
    ("Y", "I"): (1, "Y"),
    ("Z", "I"): (1, "Z"),
    ("X", "X"): (1, "I"),
    ("Y", "Y"): (1, "I"),
    ("Z", "Z"): (1, "I"),
    ("X", "Y"): (1j, "Z"),
    ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"),
    ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"),
    ("X", "Z"): (-1j, "Y"),
}


def _compose_strings(s1: str, s2: str) -> tuple[complex, str]:
    """Site-by-site operator product (matrix multiplication) of two equal-length Pauli strings."""
    phase: complex = 1
    chars = []
    for a, b in zip(s1, s2):
        local_phase, char = _PAULI_MULT[(a, b)]
        phase *= local_phase
        chars.append(char)
    return phase, "".join(chars)


class PauliString(Observable):
    """Weighted tensor product of single-qubit Paulis, e.g. PauliString("XZI", coeff=0.5).
    Indices are [0, ..., len(string)-1]. Accepts upper or lower case.
    """

    def __init__(self, string: str, coeff: complex = 1.0):
        self.string = string.upper()
        if any(c not in PAULI_MATRICES for c in self.string):
            raise ValueError(f"Invalid Pauli characters in '{string}'; use I, X, Y, Z.")
        self.coeff = complex(coeff)
        n = len(self.string)
        self.num_sites = n
        super().__init__(f"P({self.string})", list(range(n)))

    @property
    def matrix(self) -> np.ndarray:
        mats = [PAULI_MATRICES[c] for c in self.string]
        return self.coeff * reduce(np.kron, mats)

    def as_pauli_sum(self) -> PauliSum:
        return PauliSum([self])

    def compose(self, other: PauliString | PauliSum) -> PauliString | PauliSum:
        """Operator product self @ other (matrix multiplication, NOT tensor product). Same length required."""
        if isinstance(other, PauliSum):
            return PauliSum([self]).compose(other)
        if len(self.string) != len(other.string):
            raise ValueError(
                f"compose requires equal-length strings; got {len(self.string)} and {len(other.string)}."
            )
        phase, string = _compose_strings(self.string, other.string)
        return PauliString(string, self.coeff * other.coeff * phase)

    def __matmul__(self, other: PauliString | PauliSum) -> PauliString | PauliSum:
        return self.compose(other)

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

    def compose(self, other: PauliString | PauliSum) -> PauliSum:
        """Operator product self @ other (matrix multiplication). Distributes over both operands' terms."""
        other_terms = other.terms if isinstance(other, PauliSum) else [other]
        return PauliSum([s.compose(o) for s in self.terms for o in other_terms])

    def __matmul__(self, other: PauliString | PauliSum) -> PauliSum:
        return self.compose(other)

    def simplify(self, atol: float = 1e-12) -> PauliSum:
        """Merge terms with identical strings (sum coefficients); drop near-zero terms."""
        merged: dict[str, complex] = {}
        for term in self.terms:
            merged[term.string] = merged.get(term.string, 0) + term.coeff
        kept = [
            PauliString(string, coeff)
            for string, coeff in merged.items()
            if abs(coeff) > atol
        ]
        if not kept:
            kept = [PauliString("I" * self.num_sites, 0.0)]
        return PauliSum(kept)

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
    figure out how to contract
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
