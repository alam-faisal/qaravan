"""Hamiltonian base class, HamiltonianTerm, TFI, Heisenberg1D, FermiHubbard1D."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh

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

if TYPE_CHECKING:
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
        """Embed all local terms into the full n-qubit space, sum, and simplify.

        Simplifying (merging identical strings) matters in practice: a JW-encoded
        fermionic Hamiltonian's onsite terms each individually pad out to an all-identity
        string once embedded, so a Hamiltonian with many onsite terms accumulates many
        separate copies of the same string if they're never merged -- pure waste feeding
        into as_matrix()'s dense per-string matrix construction. Simplifying never changes
        .matrix numerically; it just avoids redundant work and gives a smaller __repr__.
        """
        embedded: list[PauliString] = []
        for term in self.terms:
            embedded.extend(
                _embed_local_obs(term.local_obs, term.sites, self.num_sites)
            )
        return PauliSum(embedded).simplify()

    def as_matrix(self) -> np.ndarray:
        """Full 2^n × 2^n Hamiltonian matrix."""
        return self.as_observable().matrix

    def ground_state(
        self,
        method: str = "exact",
        fixed_particle_number: int | None = None,
    ) -> Statevector:
        """Ground state. Returns Statevector.

        method="exact" (default): full dense diagonalization (numpy.linalg.eigh). Computes the
        whole spectrum even though only the bottom eigenvector is kept -- O(n^3), but robust at
        any size, including the tiny systems (n<20) most tests use.
        method="lanczos": scipy.sparse.linalg.eigsh(k=1) -- only the extremal eigenpair,
        O(n^2) per iteration instead of diagonalizing everything. Much faster once n is large
        enough that "compute everything" is wasteful; raises ValueError if the (possibly
        sector-restricted) matrix is too small for that to make sense -- use "exact" there.
        For a degenerate ground level, either method returns *some* vector spanning the ground
        subspace, not necessarily the same one run to run.

        fixed_particle_number restricts to the popcount-k sector of the computational basis
        before diagonalizing -- meaningful whenever the Hamiltonian conserves total
        occupation number, e.g. any Jordan-Wigner-encoded fermionic Hamiltonian.
        """
        # Local import: core/ must not depend on backends/ at module load time -- backends
        # import core.base, so a top-level import here is a circular-import trap that only
        # bites depending on which module a caller happens to import first.
        from qaravan.backends.statevector import Statevector

        H = self.as_matrix()
        if fixed_particle_number is None:
            sector = None
            H_reduced = H
        else:
            sector = [
                i
                for i in range(2**self.num_sites)
                if bin(i).count("1") == fixed_particle_number
            ]
            H_reduced = H[np.ix_(sector, sector)]

        if method == "exact":
            _, evecs = np.linalg.eigh(H_reduced)
            ground_vec = evecs[:, 0]
        elif method == "lanczos":
            n = H_reduced.shape[0]
            ncv = min(n - 1, max(3, 20))
            if ncv <= 1:
                raise ValueError(
                    f"method='lanczos' needs a larger matrix than {n}x{n}; use method='exact'."
                )
            _, evecs = eigsh(H_reduced, k=1, which="SA", ncv=ncv)
            ground_vec = evecs[:, 0]
        else:
            raise ValueError(
                f"Unknown method {method!r}; expected 'exact' or 'lanczos'."
            )

        if sector is None:
            return Statevector(array=ground_vec)
        full_vec = np.zeros(2**self.num_sites, dtype=complex)
        for local_idx, global_idx in enumerate(sector):
            full_vec[global_idx] = ground_vec[local_idx]
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

    def __repr__(self) -> str:
        return repr(self.as_observable())


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


def interleaved_mode_order(
    num_fermions: int, annotate: bool = False
) -> tuple[int, ...] | tuple[tuple[int, ...], str]:
    """mode_order for FermiHubbard1D's interleaved layout: qubit 2j <- up_j, qubit 2j+1 <- down_j.

    annotate=True also returns a human-readable qubit-layout string, e.g.
    'q0=↑0 q1=↓0 q2=↑1 q3=↓1', as (mode_order, annotation).
    """
    order = [0] * (2 * num_fermions)
    for j in range(num_fermions):
        order[2 * j], order[2 * j + 1] = j, num_fermions + j
    order = tuple(order)
    if not annotate:
        return order

    def label(mode: int) -> str:
        return f"↑{mode}" if mode < num_fermions else f"↓{mode - num_fermions}"

    annotation = " ".join(f"q{q}={label(mode)}" for q, mode in enumerate(order))
    return order, annotation


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
# AndersonImpurity
# ---------------------------------------------------------------------------


def _aim_onsite_terms(
    eps_d: float, U: float, eps_levels: list[float], num_bath: int, mode
) -> list:
    """eps_d/U impurity terms plus per-level (bath or chain) on-site number terms."""
    onsite = [eps_d * number(mode(0, 0)), eps_d * number(mode(0, 1))]
    onsite += density_density(mode(0, 0), mode(0, 1), U)
    for k in range(1, num_bath + 1):
        level = eps_levels[k - 1]
        onsite += [level * number(mode(k, 0)), level * number(mode(k, 1))]
    return onsite


def _aim_star_terms(
    eps_d: float, U: float, eps_bath: list[float], V: list[float]
) -> tuple[list[int], list]:
    """Flat AIM (star) FermionTerms, plus the OUTPUT HamiltonianTerm count of each Trotter
    layer: one layer per bath index (every hybridization bond touches the impurity, so none
    of them commute with each other), plus a final onsite layer. mode(site, spin) =
    spin*(num_bath+1) + site, site 0 = impurity, site k = bath k.
    """
    if len(eps_bath) != len(V):
        raise ValueError(
            f"eps_bath and V must have the same length; got {len(eps_bath)} and {len(V)}."
        )
    num_bath = len(eps_bath)

    def mode(site: int, spin: int) -> int:
        return spin * (num_bath + 1) + site

    hyb_layers = []
    for k in range(1, num_bath + 1):
        layer = hopping(mode(0, 0), mode(k, 0), V[k - 1]) + hopping(
            mode(0, 1), mode(k, 1), V[k - 1]
        )
        hyb_layers.append(layer)

    onsite = _aim_onsite_terms(eps_d, U, eps_bath, num_bath, mode)

    layer_sizes = [len(layer) // 2 for layer in hyb_layers] + [len(onsite)]
    terms = [term for layer in hyb_layers for term in layer] + onsite
    return layer_sizes, terms


def _aim_chain_terms(
    eps_d: float, U: float, eps_chain: list[float], t_chain: list[float], V: float
) -> tuple[list[int], list]:
    """Flat AIM (chain) FermionTerms, plus per-layer OUTPUT HamiltonianTerm counts (up-even,
    up-odd, down-even, down-odd, onsite) -- structurally identical to _fh_terms, since the
    impurity is just site 0 of an ordinary open chain of length num_bath+1, with V taking the
    place of a uniform hopping for the impurity<->first-site bond.
    """
    num_bath = len(eps_chain)
    expected_t_len = max(num_bath - 1, 0)
    if len(t_chain) != expected_t_len:
        raise ValueError(
            f"t_chain must have length num_bath-1={expected_t_len}; got {len(t_chain)}."
        )

    def mode(site: int, spin: int) -> int:
        return spin * (num_bath + 1) + site

    bonds = (
        LinearLattice(num_bath + 1, periodic=False).nn_pairs() if num_bath >= 1 else []
    )
    couplings = [V, *t_chain]
    even_bonds, even_couplings = bonds[::2], couplings[::2]
    odd_bonds, odd_couplings = bonds[1::2], couplings[1::2]

    def hop_layer(bond_list, coupling_list, spin):
        return [
            term
            for (a, b), coeff in zip(bond_list, coupling_list)
            for term in hopping(mode(a, spin), mode(b, spin), coeff)
        ]

    up_even = hop_layer(even_bonds, even_couplings, 0)
    up_odd = hop_layer(odd_bonds, odd_couplings, 0)
    down_even = hop_layer(even_bonds, even_couplings, 1)
    down_odd = hop_layer(odd_bonds, odd_couplings, 1)

    onsite = _aim_onsite_terms(eps_d, U, eps_chain, num_bath, mode)

    layer_sizes = [
        len(even_bonds),
        len(odd_bonds),
        len(even_bonds),
        len(odd_bonds),
        len(onsite),
    ]
    chunks = [up_even, up_odd, down_even, down_odd, onsite]
    return layer_sizes, [term for chunk in chunks for term in chunk]


def bath_interleaved_mode_order(
    num_bath: int, annotate: bool = False
) -> tuple[int, ...] | tuple[tuple[int, ...], str]:
    """mode_order placing bath modes first (interleaved by spin: bath1_up, bath1_down,
    bath2_up, bath2_down, ...), impurity last: qubits [b1up,b1down,...,bNup,bNdown,i_up,i_down].

    annotate=True also returns a human-readable qubit-layout string, e.g.
    'q0=b1↑ q1=b1↓ q2=b2↑ q3=b2↓ q4=i↑ q5=i↓', as (mode_order, annotation).
    """
    order = []
    for k in range(1, num_bath + 1):
        order.append(k)
        order.append(num_bath + 1 + k)
    order.append(0)
    order.append(num_bath + 1)
    order = tuple(order)
    if not annotate:
        return order

    def label(mode: int) -> str:
        spin_symbol = "↑" if mode < num_bath + 1 else "↓"
        site = mode % (num_bath + 1)
        return f"i{spin_symbol}" if site == 0 else f"b{site}{spin_symbol}"

    annotation = " ".join(f"q{q}={label(mode)}" for q, mode in enumerate(order))
    return order, annotation


class AndersonImpurity(FermionOperator):
    """Single-orbital Anderson impurity model (star or chain geometry). Bare -- no qubit
    encoding until .encode(encoding) is called, exactly like FermiHubbard1D.

    Build via .star(...) or .chain(...); the plain constructor is for their internal use --
    the two geometries take different-shaped parameters and there is no conversion between
    them (the star<->chain Lanczos mapping is not implemented here).

    Canonical mode index: spin*(num_bath+1) + site, site 0 = impurity, sites 1..num_bath =
    bath/chain sites 1..num_bath (spin 0=up, 1=down). Total modes/qubits: 2*(num_bath+1).
    """

    def __init__(
        self,
        num_bath: int,
        eps_d: float,
        U: float,
        topology: str,
        terms: list,
        layer_sizes: list[int],
    ):
        self.num_bath = num_bath
        self.eps_d = eps_d
        self.U = U
        self.topology = topology
        self._layer_sizes = layer_sizes
        super().__init__(terms, 2 * (num_bath + 1))

    @classmethod
    def star(
        cls, eps_d: float, U: float, eps_bath: list[float], V: list[float]
    ) -> AndersonImpurity:
        """H = Himp + Hbath + Hhyb: impurity coupled directly to each of len(eps_bath) bath
        modes via its own V_k; bath modes never couple to each other."""
        layer_sizes, terms = _aim_star_terms(eps_d, U, eps_bath, V)
        return cls(len(eps_bath), eps_d, U, "star", terms, layer_sizes)

    @classmethod
    def chain(
        cls,
        eps_d: float,
        U: float,
        eps_chain: list[float],
        t_chain: list[float],
        V: float,
    ) -> AndersonImpurity:
        """H = Himp + chain-mapped Hbath+Hhyb: impurity coupled only to the first of
        len(eps_chain) nearest-neighbor-hopping chain sites, via a single V."""
        layer_sizes, terms = _aim_chain_terms(eps_d, U, eps_chain, t_chain, V)
        return cls(len(eps_chain), eps_d, U, "chain", terms, layer_sizes)

    def encode(self, encoding: FermionicEncoding) -> Hamiltonian:
        """Same pattern as FermiHubbard1D.encode(): the generic, Trotter-agnostic
        FermionOperator.encode() runs first; Trotter-layer structure (star: one layer per
        bath index; chain: even/odd bond splitting) is this model's own business, re-chunked
        onto the result afterward."""
        ham = super().encode(encoding)
        ham.trotter_groups = _chunk(ham.terms, self._layer_sizes)
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
