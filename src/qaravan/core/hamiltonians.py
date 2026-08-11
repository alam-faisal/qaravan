"""Hamiltonian base class, HamiltonianTerm, TFI, Heisenberg1D, FermiHubbard1D."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from qaravan.core.base import Observable
from qaravan.core.circuit import Circuit
from qaravan.core.fermions import (
    FermionOperator,
    FermionicEncoding,
    density_density,
    hopping,
    number,
)
from qaravan.core.gates import MatrixGate
from qaravan.core.lattices import LinearLattice
from qaravan.core.observables import PauliString, PauliSum
from qaravan.backends.statevector import Statevector


@dataclass
class HamiltonianTerm:
    """A local interaction: an Observable on len(sites) qubits, pinned to specific sites in the full system.

    Kept separate from PauliString/PauliSum deliberately. PauliString/PauliSum are
    full-system observables (string length == system size) used for expectation values,
    matrix construction, and measurement. Merging site-pinning into them would require
    adding an indices parameter, relaxing PauliSum's same-length validation, and
    patching _expectation_pauli_string — complicating a clean, well-tested abstraction
    for a use case (Hamiltonian construction) that is better served by a dedicated type.

    local_obs lives in the SMALL space of the term itself (1- or 2-qubit).
    sites records where in the full system it acts — needed for Trotter gate placement
    and for I-padding when building the global PauliSum in as_observable().
    local_obs must be a PauliString or PauliSum for as_observable() to work.
    """

    sites: list[int]
    local_obs: Observable


def _embed_local_obs(obs: Observable, sites: list[int], n: int) -> list[PauliString]:
    """Embed a local PauliString or PauliSum into the n-qubit space by I-padding.

    sites must match the number of qubits in obs. E.g. PauliString("ZZ"), sites=[1,2], n=4 → ["IZZI"].
    """
    if isinstance(obs, PauliSum):
        embedded = []
        for ps in obs.terms:
            embedded.extend(_embed_local_obs(ps, sites, n))
        return embedded
    if isinstance(obs, PauliString):
        chars = ["I"] * n
        for site, op in zip(sites, obs.string):
            chars[site] = op
        return [PauliString("".join(chars), obs.coeff)]
    raise TypeError(
        f"local_obs must be PauliString or PauliSum; got {type(obs).__name__}"
    )


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
        _gate_fusion_applied: bool = True,
    ):
        self.num_sites = num_sites
        self.terms = terms
        self.trotter_groups = trotter_groups
        # Set by FermionOperator.encode() for generically-encoded Hamiltonians, where no
        # pass has merged commuting same-qubit terms into fewer gates. Hand-built models
        # (TFI, Heisenberg1D, ...) don't pass this and get no warning.
        self._gate_fusion_applied = _gate_fusion_applied

    def as_observable(self) -> PauliSum:
        """Embed all local terms into the full n-qubit space and sum."""
        embedded: list[PauliString] = []
        for term in self.terms:
            embedded.extend(
                _embed_local_obs(term.local_obs, term.sites, self.num_sites)
            )
        return PauliSum(embedded)

    def as_matrix(self) -> np.ndarray:
        """Full 2^n × 2^n Hamiltonian matrix."""
        return self.as_observable().matrix

    def ground_state(
        self,
        method: str = "exact",
        fixed_particle_number: int | None = None,
    ) -> Statevector:
        """Ground state via exact diagonalization (numpy.linalg.eigh). Returns Statevector.

        fixed_particle_number restricts to the popcount-k sector of the computational basis
        before diagonalizing -- meaningful whenever the Hamiltonian conserves total
        occupation number, e.g. any Jordan-Wigner-encoded fermionic Hamiltonian.
        """
        H = self.as_matrix()
        if fixed_particle_number is None:
            evals, evecs = np.linalg.eigh(H)
            _ = evals  # ascending; evecs[:,0] is the ground state
            return Statevector(array=evecs[:, 0])

        sector = [
            i
            for i in range(2**self.num_sites)
            if bin(i).count("1") == fixed_particle_number
        ]
        H_sector = H[np.ix_(sector, sector)]
        evals, evecs = np.linalg.eigh(H_sector)
        _ = evals
        full_vec = np.zeros(2**self.num_sites, dtype=complex)
        for local_idx, global_idx in enumerate(sector):
            full_vec[global_idx] = evecs[local_idx, 0]
        return Statevector(array=full_vec)

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
        if not self._gate_fusion_applied:
            warnings.warn(
                "This Hamiltonian's terms were produced by a generic fermion-to-qubit "
                "encoding with no gate-fusion optimization pass -- the Trotter circuit may "
                "use more gates than necessary (e.g. commuting diagonal terms on the same "
                "qubits are not merged into one gate). Correctness and Trotter error "
                "scaling are unaffected.",
                stacklevel=2,
            )
        if self.num_sites <= 2:
            raise ValueError(
                f"trotter_circuit requires num_sites > 2; got {self.num_sites}."
            )
        groups = [g for g in self.trotter_groups if g]  # skip empty groups

        if order == 1:
            gates = _trotter_order1(groups, dt)
        elif order == 2:
            gates = _trotter_order2(groups, dt)
        else:
            raise ValueError(f"Trotter order must be 1 or 2; got {order}.")

        return Circuit(gates, num_sites=self.num_sites)


def _trotter_order1(groups: list[list[HamiltonianTerm]], dt: float) -> list[MatrixGate]:
    """Apply each group once at step size dt."""
    gates: list[MatrixGate] = []
    for group in groups:
        for term in group:
            mat = expm(-1j * dt * term.local_obs.matrix)
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
            mat = expm(-1j * half_dt * term.local_obs.matrix)
            gates.append(MatrixGate("Trotter", term.sites, mat))
    # last group: full step
    for term in groups[-1]:
        mat = expm(-1j * dt * term.local_obs.matrix)
        gates.append(MatrixGate("Trotter", term.sites, mat))
    # backward half-steps: reverse of all but last group
    for group in reversed(groups[:-1]):
        for term in group:
            mat = expm(-1j * half_dt * term.local_obs.matrix)
            gates.append(MatrixGate("Trotter", term.sites, mat))
    return gates


# ---------------------------------------------------------------------------
# FermiHubbard1D
# ---------------------------------------------------------------------------


def _fh_terms(
    L: int, t: float, U: float, mu: float, periodic: bool
) -> tuple[list[int], list]:
    """Flat FH FermionTerms in canonical mode order (mode = spin*L + site, spin 0=up,
    1=down), plus the OUTPUT HamiltonianTerm count of each of the 5 Trotter layers
    (up-even, up-odd, down-even, down-odd, onsite) in the SAME order they appear in the
    flat list -- so the qubit-space Hamiltonian's terms can be re-chunked into
    trotter_groups after encoding, with no re-derivation. Hopping bonds contribute 2
    FermionTerms each but 1 output HamiltonianTerm (jordan_wigner re-pairs the h.c. halves);
    onsite terms are each individually self-adjoint, 1-to-1. See docs/jw_fermi_hubbard_1d.md.
    """
    bonds = LinearLattice(L, periodic=periodic).nn_pairs()
    even_bonds, odd_bonds = bonds[::2], bonds[1::2]

    up_even = [term for a, b in even_bonds for term in hopping(a, b, -t)]
    up_odd = [term for a, b in odd_bonds for term in hopping(a, b, -t)]
    down_even = [term for a, b in even_bonds for term in hopping(L + a, L + b, -t)]
    down_odd = [term for a, b in odd_bonds for term in hopping(L + a, L + b, -t)]
    onsite = [
        term
        for j in range(L)
        for term in [mu * number(j), mu * number(L + j), *density_density(j, L + j, U)]
    ]

    layer_output_sizes = [
        len(even_bonds),
        len(odd_bonds),
        len(even_bonds),
        len(odd_bonds),
        3 * L,
    ]
    chunks = [up_even, up_odd, down_even, down_odd, onsite]
    return layer_output_sizes, [term for chunk in chunks for term in chunk]


def interleaved_mode_order(num_fermions: int) -> tuple[int, ...]:
    """mode_order for FermiHubbard1D's interleaved layout: qubit 2j <- up_j, qubit 2j+1 <- down_j."""
    order = [0] * (2 * num_fermions)
    for j in range(num_fermions):
        order[2 * j], order[2 * j + 1] = j, num_fermions + j
    return tuple(order)


def _chunk(items: list, sizes: list[int]) -> list[list]:
    """Split a flat list into consecutive chunks of the given sizes."""
    out, offset = [], 0
    for size in sizes:
        out.append(items[offset : offset + size])
        offset += size
    return out


class FermiHubbard1D(FermionOperator):
    """1D Fermi-Hubbard model: H = -t Σ_{j,σ}(c†c + h.c.) + U Σ_j n↑n↓ + μ Σ_{j,σ} n_{j,σ}.

    Bare -- no qubit encoding until .encode(encoding) is called. Canonical mode index is
    spin*num_fermions + site (spin 0=up, 1=down), chosen so that FermionicEncoding's default
    (mode_order=None) reproduces the "stacked" register [↑_0..↑_{L-1}, ↓_0..↓_{L-1}];
    "interleaved" ([↑_0,↓_0,↑_1,↓_1,...]) is FermionicEncoding(..., mode_order=
    interleaved_mode_order(num_fermions)). Total modes/qubits: 2 * num_fermions.
    See docs/jw_fermi_hubbard_1d.md for derivations and hardware tradeoffs.
    """

    def __init__(
        self,
        num_fermions: int,
        t: float = 1.0,
        U: float = 1.0,
        mu: float = 0.0,
        periodic: bool = False,
    ):
        if periodic and num_fermions < 2:
            raise ValueError("periodic=True requires num_fermions >= 2.")
        self.num_fermions = num_fermions
        self.t = t
        self.U = U
        self.mu = mu
        self.periodic = periodic
        self._layer_output_sizes, terms = _fh_terms(num_fermions, t, U, mu, periodic)
        super().__init__(terms, 2 * num_fermions)

    def encode(self, encoding: FermionicEncoding) -> Hamiltonian:
        """Trotter-layer structure (which terms mutually commute) is FH's own business --
        computed once at construction, re-chunked onto whatever Hamiltonian the generic,
        Trotter-agnostic FermionOperator.encode() produces."""
        ham = super().encode(encoding)
        ham.trotter_groups = _chunk(ham.terms, self._layer_output_sizes)
        return ham


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
            HamiltonianTerm([i, j], PauliString("ZZ", -J))
            for i, j in lattice.nn_pairs()
        ]
        field_terms = [HamiltonianTerm([i], PauliString("X", -h)) for i in range(n)]

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
        field_terms = [HamiltonianTerm([i], PauliString("Z", -h)) for i in range(n)]

        even_bonds = bond_terms[::2]
        odd_bonds = bond_terms[1::2]
        trotter_groups = [even_bonds, odd_bonds, field_terms]

        super().__init__(n, bond_terms + field_terms, trotter_groups)
