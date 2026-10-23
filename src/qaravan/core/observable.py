"""Concrete Observable subclasses."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from qaravan.core.base import Observable
from qaravan.core.pauli import PauliString, PauliSum

# === Local Hamiltonian terms ===


@dataclass
class HamiltonianTerm:
    """Local PauliString or PauliSum pinned to sites for deferred full-register embedding."""

    sites: list[int]
    local_obs: Observable


def _embed_local_obs(
    obs: Observable, sites: list[int], num_sites: int
) -> list[PauliString]:
    """I-pad a local PauliString or PauliSum onto sites in a num_sites register."""
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
    """Embed local terms and merge duplicate full-register Pauli strings."""
    embedded: list[PauliString] = []
    for term in terms:
        embedded.extend(_embed_local_obs(term.local_obs, term.sites, num_sites))
    if not embedded:  # every coupling was zero
        return PauliSum([PauliString("I" * num_sites, 0.0)])
    return PauliSum(embedded).simplify()


def restrict_pauli_sum(pauli_sum: PauliSum) -> HamiltonianTerm:
    """Restrict a full-register PauliSum to its nontrivial sites."""
    touched_sites = set()
    for term in pauli_sum.terms:
        touched_sites.update(i for i, ch in enumerate(term.string) if ch != "I")
    sites = sorted(touched_sites)
    local_terms = [
        PauliString("".join(term.string[i] for i in sites), term.coeff)
        for term in pauli_sum.terms
    ]
    return HamiltonianTerm(sites, PauliSum(local_terms))


# === Local and aggregate observables ===


class LocalOp(Observable):
    """Dense local operator returned without full-register embedding."""

    def __init__(self, mat: np.ndarray, indices: int | list[int]):
        self._matrix = np.asarray(mat, dtype=complex)
        super().__init__("LocalOp", indices)

    def to_matrix(self) -> np.ndarray:
        return self._matrix


class Magnetization(PauliSum):
    """Mean single-site Pauli (1/n) Σ_i σ_i^axis."""

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
    """Particle count Σ_q (I-Z_q)/2 over all or selected qubits."""

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
