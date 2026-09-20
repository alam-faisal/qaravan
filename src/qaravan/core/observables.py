"""Concrete Observable subclasses"""

from __future__ import annotations
from dataclasses import dataclass
from functools import reduce
import numpy as np
from qaravan.core._repr import (
    as_math,
    elide,
    format_coefficient,
    join_terms,
    latex_pauli_string,
    latex_term,
    math_more,
)
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

    def to_matrix(self) -> np.ndarray:
        mats = [PAULI_MATRICES[c] for c in self.string]
        return self.coeff * reduce(np.kron, mats)

    def bitmasks(self) -> tuple[int, int, int]:
        """(x_mask, z_mask, n_Y): P|j⟩ = coeff · i^n_Y · (-1)^popcount(j & z_mask) |j ^ x_mask⟩.

        A Pauli string is a generalized permutation matrix, so these three integers carry the
        whole of it. Qubit q is bit num_sites-1-q of the basis index, matching to_matrix's
        kron order.
        """
        n = self.num_sites
        x_mask = sum(1 << (n - 1 - q) for q, c in enumerate(self.string) if c in "XY")
        z_mask = sum(1 << (n - 1 - q) for q, c in enumerate(self.string) if c in "YZ")
        return x_mask, z_mask, self.string.count("Y")

    def as_pauli_sum(self) -> PauliSum:
        return PauliSum([self])

    def compose(self, other: PauliString) -> PauliString:
        """Operator product self @ other (matrix multiplication, NOT tensor product). Same length required."""
        if len(self.string) != len(other.string):
            raise ValueError(
                f"compose requires equal-length strings; got {len(self.string)} and {len(other.string)}."
            )
        phase, string = _compose_strings(self.string, other.string)
        return PauliString(string, self.coeff * other.coeff * phase)

    def __matmul__(self, other: PauliString) -> PauliString:
        if not isinstance(other, PauliString):
            return NotImplemented
        return self.compose(other)

    def __mul__(self, scalar: complex) -> PauliString:
        return PauliString(self.string, self.coeff * scalar)

    def __rmul__(self, scalar: complex) -> PauliString:
        return self.__mul__(scalar)

    def __add__(self, other: PauliString) -> PauliSum:
        if isinstance(other, PauliString):
            return PauliSum([self, other])
        return NotImplemented

    def __radd__(self, other: PauliString) -> PauliSum:
        return self.__add__(other)

    def __repr__(self) -> str:
        if self.coeff == 1.0:
            return self.string
        return f"{format_coefficient(self.coeff)} {self.string}"

    def _repr_latex_(self) -> str:
        return as_math(latex_term(self.coeff, latex_pauli_string(self.string)))


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

    def to_matrix(self) -> np.ndarray:
        return sum(t.to_matrix() for t in self.terms)  # type: ignore[return-value]

    def as_pauli_sum(self) -> PauliSum:
        return self

    def compose(self, other: PauliString | PauliSum) -> PauliSum:
        """Operator product self @ other (matrix multiplication). Distributes over both operands' terms."""
        other_terms = other.terms if isinstance(other, PauliSum) else [other]
        composed_terms = []
        for term in self.terms:
            for other_term in other_terms:
                composed_terms.append(term.compose(other_term))
        return PauliSum(composed_terms)

    def __matmul__(self, other: PauliString | PauliSum) -> PauliSum:
        return self.compose(other)

    def __rmatmul__(self, other: PauliString) -> PauliSum:
        """Supports ps @ pauli_sum (a bare PauliString on the left)."""
        return PauliSum([other]).compose(self)

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
        return join_terms(elide([repr(t) for t in self.terms]))

    def _repr_latex_(self) -> str:
        parts = [latex_term(t.coeff, latex_pauli_string(t.string)) for t in self.terms]
        return as_math(join_terms(elide(parts, marker=math_more)))


@dataclass
class HamiltonianTerm:
    """A local Observable pinned to sites, kept separate from PauliString/PauliSum (which are
    full-register, length == num_sites) so Hamiltonian construction can work with a small
    local operator instead of a full n-qubit one until as_observable() embeds it.

    local_obs must be a PauliString or PauliSum for as_observable() to work.
    """

    sites: list[int]
    local_obs: Observable


def _embed_local_obs(
    obs: Observable, sites: list[int], num_sites: int
) -> list[PauliString]:
    """Embed a local PauliString or PauliSum into the num_sites-qubit space by I-padding.

    sites must match the number of qubits in obs. E.g. PauliString("ZZ"), sites=[1,2],
    num_sites=4 -> ["IZZI"].
    """
    if isinstance(obs, PauliSum):
        embedded = []
        for term in obs.terms:
            embedded.extend(_embed_local_obs(term, sites, num_sites))
        return embedded
    if isinstance(obs, PauliString):
        chars = ["I"] * num_sites
        for site, op in zip(sites, obs.string):
            chars[site] = op
        return [PauliString("".join(chars), obs.coeff)]
    raise TypeError(
        f"local_obs must be PauliString or PauliSum; got {type(obs).__name__}"
    )


def embed_hamiltonian_terms(terms: list[HamiltonianTerm], num_sites: int) -> PauliSum:
    """Embed all local HamiltonianTerms into the full num_sites-qubit space and simplify
    (merge duplicate strings) -- doesn't change the matrix, just avoids redundant terms."""
    embedded: list[PauliString] = []
    for term in terms:
        embedded.extend(_embed_local_obs(term.local_obs, term.sites, num_sites))
    if not embedded:  # every coupling was zero
        return PauliSum([PauliString("I" * num_sites, 0.0)])
    return PauliSum(embedded).simplify()


def restrict_pauli_sum(pauli_sum: PauliSum) -> HamiltonianTerm:
    """Inverse of embed_hamiltonian_terms: restrict a full-register PauliSum down to a
    HamiltonianTerm covering just the sites it touches."""
    touched_sites = set()
    for term in pauli_sum.terms:
        touched_sites.update(i for i, ch in enumerate(term.string) if ch != "I")
    sites = sorted(touched_sites)
    local_terms = [
        PauliString("".join(term.string[i] for i in sites), term.coeff)
        for term in pauli_sum.terms
    ]
    return HamiltonianTerm(sites, PauliSum(local_terms))


class LocalOp(Observable):
    """Generic local Hermitian operator on specified sites.

    matrix is returned as-is (no embedding). Backends use obs.indices to
    figure out how to contract
    """

    def __init__(self, mat: np.ndarray, indices: int | list[int]):
        self._matrix = np.asarray(mat, dtype=complex)
        super().__init__("LocalOp", indices)

    def to_matrix(self) -> np.ndarray:
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


class NumberOperator(PauliSum):
    """
    n_q = (I - Z_q)/2, eigenvalue 1 on |1> and 0 on |0>. Unnormalized -- it counts particles.
    if qubits is passed, it counts only those qubits; otherwise it counts all n qubits.
    """

    def __init__(self, n: int, qubits: list[int] | None = None):
        self.n = n
        self.qubits = list(range(n)) if qubits is None else list(qubits)
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"qubits must not repeat; got {self.qubits}.")
        if not all(0 <= qubit < n for qubit in self.qubits):
            raise ValueError(f"qubits must lie in [0, {n}); got {self.qubits}.")
        identity = PauliString("I" * n, 0.5 * len(self.qubits))
        z_terms = [
            PauliString("I" * qubit + "Z" + "I" * (n - qubit - 1), -0.5)
            for qubit in self.qubits
        ]
        super().__init__([identity, *z_terms])
        self.name = "NumberOperator"
