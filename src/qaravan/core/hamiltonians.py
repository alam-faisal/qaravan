"""Hamiltonian base class, HamiltonianTerm, TFI, Heisenberg1D, FermiHubbard1D."""

from dataclasses import dataclass
from typing import Self

import numpy as np

from qaravan.core.base import Observable
from qaravan.core.fermions import (
    FermionicEncoding,
    FermionicSum,
    density_density,
    hopping,
    number,
)
from qaravan.core.lattices import LinearLattice
from qaravan.core.observables import PauliString, PauliSum


@dataclass
class HamiltonianTerm:
    """A local Observable pinned to sites, kept separate from PauliString/PauliSum (which are
    full-register, length == num_sites) so Hamiltonian construction can work with a small
    local operator instead of a full n-qubit one until as_observable() embeds it.

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


@dataclass
class Hamiltonian:
    """Generic Hamiltonian: a list of local HamiltonianTerms with a declared Trotter grouping.

    trotter_groups is a partition of terms into sublists. Within each sublist all terms
    act on disjoint sites (so they commute and form a single Trotter layer).
    """

    num_sites: int
    terms: list[HamiltonianTerm]
    trotter_groups: list[list[HamiltonianTerm]] | None = None

    def as_observable(self) -> PauliSum:
        """Embed all local terms into the full n-qubit space and simplify (merge duplicate
        strings) -- doesn't change .matrix, just avoids redundant terms."""
        embedded: list[PauliString] = []
        for term in self.terms:
            embedded.extend(
                _embed_local_obs(term.local_obs, term.sites, self.num_sites)
            )
        return PauliSum(embedded).simplify()

    def as_matrix(self) -> np.ndarray:
        """Full 2^n × 2^n Hamiltonian matrix."""
        return self.as_observable().matrix

    def __repr__(self) -> str:
        return repr(self.as_observable())


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


# ---------------------------------------------------------------------------
# Fermionic Hamiltonians: FermionicHamiltonian, FermiHubbard1D, AndersonImpurity
# ---------------------------------------------------------------------------


class FermionicHamiltonian(FermionicSum):
    """Hermitian-asserted FermionicSum -- what FermiHubbard1D/AndersonImpurity actually are."""

    def __init__(self, terms: list, num_modes: int):
        super().__init__(terms, num_modes)
        if not self.is_hermitian():
            raise ValueError(
                f"{type(self).__name__} is not Hermitian -- check the terms passed in."
            )


def _chunk(items: list, sizes: list[int]) -> list[list]:
    """Split a flat list into consecutive chunks of the given sizes."""
    out, offset = [], 0
    for size in sizes:
        out.append(items[offset : offset + size])
        offset += size
    return out


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


class FermiHubbard1D(FermionicHamiltonian):
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
        Trotter-agnostic FermionicSum.encode() produces."""
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


class AndersonImpurity(FermionicHamiltonian):
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
    ) -> Self:
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
    ) -> Self:
        """H = Himp + chain-mapped Hbath+Hhyb: impurity coupled only to the first of
        len(eps_chain) nearest-neighbor-hopping chain sites, via a single V."""
        layer_sizes, terms = _aim_chain_terms(eps_d, U, eps_chain, t_chain, V)
        return cls(len(eps_chain), eps_d, U, "chain", terms, layer_sizes)

    def encode(self, encoding: FermionicEncoding) -> Hamiltonian:
        """Same pattern as FermiHubbard1D.encode(): the generic, Trotter-agnostic
        FermionicSum.encode() runs first; Trotter-layer structure (star: one layer per
        bath index; chain: even/odd bond splitting) is this model's own business, re-chunked
        onto the result afterward."""
        ham = super().encode(encoding)
        ham.trotter_groups = _chunk(ham.terms, self._layer_sizes)
        return ham
