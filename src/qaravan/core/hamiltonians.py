"""Hamiltonian base class, HamiltonianTerm, TFI, Heisenberg1D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from qaravan.core.circuit import Circuit
from qaravan.core.gates import MatrixGate
from qaravan.core.lattices import LinearLattice
from qaravan.core.observables import PauliString, PauliSum


@dataclass
class HamiltonianTerm:
    """A local interaction: a PauliSum on len(sites) qubits, pinned to specific sites in the full system.

    Kept separate from PauliString/PauliSum deliberately. PauliString/PauliSum are
    full-system observables (string length == system size) used for expectation values,
    matrix construction, and measurement. Merging site-pinning into them would require
    adding an indices parameter, relaxing PauliSum's same-length validation, and
    patching _expectation_pauli_string — complicating a clean, well-tested abstraction
    for a use case (Hamiltonian construction) that is better served by a dedicated type.

    local_pauli_sum lives in the SMALL space of the term itself (1- or 2-qubit).
    sites records where in the full system it acts — needed for Trotter gate placement
    and for I-padding when building the global PauliSum in as_observable().
    """

    sites: list[int]
    local_pauli_sum: PauliSum


def _embed_pauli_string(ps: PauliString, sites: list[int], n: int) -> PauliString:
    """Embed a local PauliString into the n-qubit full-system space by padding with I's.

    sites must match len(ps.string). E.g. sites=[1,2], "ZZ", n=4 → "IZZI".
    """
    chars = ["I"] * n
    for site, op in zip(sites, ps.string):
        chars[site] = op
    return PauliString("".join(chars), ps.coeff)


class Hamiltonian:
    """Generic Hamiltonian: a list of local HamiltonianTerms with a declared Trotter grouping.

    trotter_groups is a partition of terms into sublists. Within each sublist all terms
    act on disjoint sites (so they commute and form a single Trotter layer).
    """

    def __init__(
        self,
        num_sites: int,
        terms: list[HamiltonianTerm],
        trotter_groups: list[list[HamiltonianTerm]] | None = None,
    ):
        self.num_sites = num_sites
        self.terms = terms
        self.trotter_groups = trotter_groups

    def as_observable(self) -> PauliSum:
        """Embed all local terms into the full n-qubit space and sum."""
        embedded: list[PauliString] = []
        for term in self.terms:
            for ps in term.local_pauli_sum.terms:
                embedded.append(_embed_pauli_string(ps, term.sites, self.num_sites))
        return PauliSum(embedded)

    def as_matrix(self) -> np.ndarray:
        """Full 2^n × 2^n Hamiltonian matrix."""
        return self.as_observable().matrix

    def ground_state(self, method: str = "exact"):
        """Ground state via exact diagonalization (numpy.linalg.eigh). Returns Statevector."""
        from qaravan.backends.statevector import Statevector

        evals, evecs = np.linalg.eigh(self.as_matrix())
        _ = evals  # ascending; evecs[:,0] is the ground state
        return Statevector(array=evecs[:, 0])

    def trotter_circuit(self, dt: float, order: int = 2) -> Circuit:
        """Single Trotter step of duration dt.

        order=1: sequential groups at full dt.
        order=2: symmetric (Suzuki-Trotter) with boundary half-steps.
        Raises NotImplementedError if trotter_groups is None.
        Raises ValueError for systems too small to have distinct even/odd bond layers.
        """
        if self.trotter_groups is None:
            raise NotImplementedError(
                f"{type(self).__name__} has no trotter_groups defined."
            )
        if self.num_sites <= 2:
            raise ValueError(
                f"trotter_circuit requires num_sites > 2; got {self.num_sites}."
            )
        groups = [g for g in self.trotter_groups if g]  # skip empty groups

        if order == 1:
            gates = _trotter_layer(groups, dt)
        elif order == 2:
            gates = _trotter_order2(groups, dt)
        else:
            raise ValueError(f"Trotter order must be 1 or 2; got {order}.")

        return Circuit(gates, num_sites=self.num_sites)


def _trotter_layer(groups: list[list[HamiltonianTerm]], dt: float) -> list[MatrixGate]:
    """Apply each group once at step size dt."""
    gates: list[MatrixGate] = []
    for group in groups:
        for term in group:
            mat = expm(-1j * dt * term.local_pauli_sum.matrix)
            gates.append(MatrixGate("Trotter", term.sites, mat))
    return gates


def _trotter_order2(groups: list[list[HamiltonianTerm]], dt: float) -> list[MatrixGate]:
    """Symmetric second-order Trotter for k groups.

    Pattern: g0(dt/2) g1(dt/2) ... g_{k-2}(dt/2) g_{k-1}(dt) g_{k-2}(dt/2) ... g0(dt/2)
    """
    gates: list[MatrixGate] = []
    half_dt = dt / 2.0
    # forward half-steps: all but last group
    for group in groups[:-1]:
        for term in group:
            mat = expm(-1j * half_dt * term.local_pauli_sum.matrix)
            gates.append(MatrixGate("Trotter", term.sites, mat))
    # last group: full step
    for term in groups[-1]:
        mat = expm(-1j * dt * term.local_pauli_sum.matrix)
        gates.append(MatrixGate("Trotter", term.sites, mat))
    # backward half-steps: reverse of all but last group
    for group in reversed(groups[:-1]):
        for term in group:
            mat = expm(-1j * half_dt * term.local_pauli_sum.matrix)
            gates.append(MatrixGate("Trotter", term.sites, mat))
    return gates


# ---------------------------------------------------------------------------
# TFI
# ---------------------------------------------------------------------------


class TFI(Hamiltonian):
    """Transverse-field Ising model: H = -J Σ_{<ij>} Z_i Z_j - h Σ_i X_i.

    Open boundary unless periodic=True.
    """

    def __init__(
        self,
        n: int,
        J: float = 1.0,
        h: float = 0.0,
        periodic: bool = False,
    ):
        self.n = n
        self.J = J
        self.h = h
        self.periodic = periodic

        lattice = LinearLattice(n, periodic=periodic)
        bond_terms = [
            HamiltonianTerm([i, j], PauliSum([PauliString("ZZ", -J)]))
            for i, j in lattice.nn_pairs()
        ]
        field_terms = [
            HamiltonianTerm([i], PauliSum([PauliString("X", -h)])) for i in range(n)
        ]

        even_bonds = bond_terms[::2]
        odd_bonds = bond_terms[1::2]
        trotter_groups = [even_bonds, odd_bonds, field_terms]

        super().__init__(n, bond_terms + field_terms, trotter_groups)


# ---------------------------------------------------------------------------
# Heisenberg1D
# ---------------------------------------------------------------------------


class Heisenberg1D(Hamiltonian):
    """1D Heisenberg chain: H = J Σ_{<ij>} (X_iX_j + Y_iY_j + Z_iZ_j) - h Σ_i Z_i.

    Bond gate is e^{-i dt J (XX+YY+ZZ)}: a single exact 4×4 gate, not decomposed into
    separate RXX/RYY/RZZ rotations (which would introduce O(dt²) intra-bond Trotter error).
    Open boundary unless periodic=True.
    """

    def __init__(
        self,
        n: int,
        J: float = 1.0,
        h: float = 0.0,
        periodic: bool = False,
    ):
        self.n = n
        self.J = J
        self.h = h
        self.periodic = periodic

        lattice = LinearLattice(n, periodic=periodic)
        bond_terms = [
            HamiltonianTerm(
                [i, j],
                PauliSum(
                    [PauliString("XX", J), PauliString("YY", J), PauliString("ZZ", J)]
                ),
            )
            for i, j in lattice.nn_pairs()
        ]
        field_terms = [
            HamiltonianTerm([i], PauliSum([PauliString("Z", -h)])) for i in range(n)
        ]

        even_bonds = bond_terms[::2]
        odd_bonds = bond_terms[1::2]
        trotter_groups = [even_bonds, odd_bonds, field_terms]

        super().__init__(n, bond_terms + field_terms, trotter_groups)
