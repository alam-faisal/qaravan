"""Hamiltonian base class and concrete models."""

from typing import Self
import numpy as np
from qaravan.core.fermions import (
    FermionicEncoding,
    FermionicSum,
    density_density,
    hopping,
    number,
)
from qaravan.core.lattices import LinearLattice
from qaravan.core.observable import HamiltonianTerm, embed_hamiltonian_terms
from qaravan.core.pauli import PauliString, PauliSum

# === Hamiltonian ===


class Hamiltonian:
    """Local terms with optional commuting, disjoint-site Trotter groups."""

    def __init__(
        self,
        terms: list[HamiltonianTerm],
        num_sites: int,
        trotter_groups: list[list[HamiltonianTerm]] | None = None,
    ):
        self.terms = terms
        self.num_sites = num_sites
        self.trotter_groups = trotter_groups

    def as_observable(self) -> PauliSum:
        """Embed local terms and merge duplicate full-register Pauli strings."""
        return embed_hamiltonian_terms(self.terms, self.num_sites)

    def to_matrix(self) -> np.ndarray:
        """Full 2^n × 2^n Hamiltonian matrix."""
        return self.as_observable().to_matrix()

    def to_sector_matrix(self, sector: np.ndarray) -> np.ndarray:
        """This Hamiltonian restricted to the ascending basis indices in sector."""
        return self.as_observable().to_sector_matrix(sector)

    def __repr__(self) -> str:
        """Header naming the model, then the algebra it embeds to."""
        return f"{type(self).__name__}(num_sites={self.num_sites})\n{self.as_observable()!r}"

    def _repr_latex_(self) -> str:
        """Model name, then the algebra it embeds to."""
        body = self.as_observable()._repr_latex_().strip("$")
        return f"$\\text{{{type(self).__name__}}}_{{{self.num_sites}}} = {body}$"


# === Spin models ===


class TFI(Hamiltonian):
    """Transverse-field Ising model: H = -J Σ_{<ij>} Z_i Z_j - h Σ_i X_i."""

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
        bond_terms = (
            [
                HamiltonianTerm([i, j], PauliString("ZZ", -J))
                for i, j in lattice.nn_pairs()
            ]
            if J
            else []
        )
        field_terms = (
            [HamiltonianTerm([i], PauliString("X", -h)) for i in range(n)] if h else []
        )

        even_bonds = bond_terms[::2]
        odd_bonds = bond_terms[1::2]
        trotter_groups = [even_bonds, odd_bonds, field_terms]

        super().__init__(bond_terms + field_terms, n, trotter_groups)


class Heisenberg1D(Hamiltonian):
    """H = JΣ_<ij>(X_iX_j+Y_iY_j+Z_iZ_j)-hΣ_iZ_i with each bond kept as one term."""

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
        bond_terms = (
            [
                HamiltonianTerm(
                    [i, j],
                    PauliSum(
                        [
                            PauliString("XX", J),
                            PauliString("YY", J),
                            PauliString("ZZ", J),
                        ]
                    ),
                )
                for i, j in lattice.nn_pairs()
            ]
            if J
            else []
        )
        field_terms = (
            [HamiltonianTerm([i], PauliString("Z", -h)) for i in range(n)] if h else []
        )

        even_bonds = bond_terms[::2]
        odd_bonds = bond_terms[1::2]
        trotter_groups = [even_bonds, odd_bonds, field_terms]

        super().__init__(bond_terms + field_terms, n, trotter_groups)


# === Fermionic models ===


class FermionicHamiltonian(FermionicSum):
    """Hermitian FermionicSum partitioned into optional commuting Trotter groups."""

    def __init__(
        self,
        terms: list,
        num_fermionic_modes: int,
        trotter_groups: list[list] | None = None,
    ):
        super().__init__(terms, num_fermionic_modes)
        self.trotter_groups = trotter_groups
        if not self.is_hermitian():
            raise ValueError(
                f"{type(self).__name__} is not Hermitian -- check the terms passed in."
            )

    def encode(self, encoding: FermionicEncoding) -> Hamiltonian:
        """Encode each Trotter group independently; each must contain complete adjoint pairs."""
        if encoding.num_modes != self.num_fermionic_modes:
            raise ValueError(
                f"Encoding is for {encoding.num_modes} modes; operator has "
                f"{self.num_fermionic_modes}."
            )
        if self.trotter_groups is None:
            terms = encoding.encode_hamiltonian_terms(self.terms)
            return Hamiltonian(terms, encoding.num_qubits, None)
        encoded_groups = [
            encoding.encode_hamiltonian_terms(group) for group in self.trotter_groups
        ]
        terms = [term for group in encoded_groups for term in group]
        return Hamiltonian(terms, encoding.num_qubits, encoded_groups)


def interleaved_mode_order(
    num_fermionic_sites: int, annotate: bool = False
) -> tuple[int, ...] | tuple[tuple[int, ...], str]:
    """Map spin-blocked modes to q_2j=up_j, q_2j+1=down_j, optionally with a layout label."""
    order = [0] * (2 * num_fermionic_sites)
    for j in range(num_fermionic_sites):
        order[2 * j], order[2 * j + 1] = j, num_fermionic_sites + j
    order = tuple(order)
    if not annotate:
        return order

    def label(mode: int) -> str:
        return (
            f"↑{mode}"
            if mode < num_fermionic_sites
            else f"↓{mode - num_fermionic_sites}"
        )

    annotation = " ".join(f"q{q}={label(mode)}" for q, mode in enumerate(order))
    return order, annotation


class FermiHubbard1D(FermionicHamiltonian):
    """1D Fermi-Hubbard model: H = -t Σ_{j,σ}(c†c + h.c.) + U Σ_j n↑n↓ + μ Σ_{j,σ} n_{j,σ}."""

    def __init__(
        self,
        num_fermionic_sites: int,
        t: float = 1.0,
        U: float = 1.0,
        mu: float = 0.0,
        periodic: bool = False,
    ):
        if periodic and num_fermionic_sites < 2:
            raise ValueError("periodic=True requires num_fermionic_sites >= 2.")
        self.num_fermionic_sites = num_fermionic_sites
        self.t = t
        self.U = U
        self.mu = mu
        self.periodic = periodic
        groups = self._groups(num_fermionic_sites, t, U, mu, periodic)
        terms = [term for group in groups for term in group]
        super().__init__(terms, 2 * num_fermionic_sites, groups)

    @staticmethod
    def _groups(
        num_fermionic_sites: int, t: float, U: float, mu: float, periodic: bool
    ) -> list[list]:
        """Group fermionic terms as up-even, up-odd, down-even, down-odd, and onsite layers."""
        bonds = LinearLattice(num_fermionic_sites, periodic=periodic).nn_pairs()
        even_bonds, odd_bonds = bonds[::2], bonds[1::2]

        up_even = [term for a, b in even_bonds for term in hopping(a, b, -t)]
        up_odd = [term for a, b in odd_bonds for term in hopping(a, b, -t)]
        down_even = [
            term
            for a, b in even_bonds
            for term in hopping(num_fermionic_sites + a, num_fermionic_sites + b, -t)
        ]
        down_odd = [
            term
            for a, b in odd_bonds
            for term in hopping(num_fermionic_sites + a, num_fermionic_sites + b, -t)
        ]
        onsite = []
        for j in range(num_fermionic_sites):
            if mu:
                onsite += [mu * number(j), mu * number(num_fermionic_sites + j)]
            if U:
                onsite += density_density(j, num_fermionic_sites + j, U)
        return [up_even, up_odd, down_even, down_odd, onsite]


def bath_interleaved_mode_order(
    num_bath: int, annotate: bool = False
) -> tuple[int, ...] | tuple[tuple[int, ...], str]:
    """Place spin-interleaved bath modes first and impurity modes last, optionally with labels."""
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
    """Star or chain AIM with positive ε lowering levels and mode=spin·(num_bath+1)+site."""

    def __init__(
        self,
        num_bath: int,
        onsite_impurity_energy: float,
        onsite_impurity_repulsion: float,
        topology: str,
        terms: list,
        groups: list[list],
    ):
        self.num_bath = num_bath
        self.onsite_impurity_energy = onsite_impurity_energy
        self.onsite_impurity_repulsion = onsite_impurity_repulsion
        self.topology = topology
        super().__init__(terms, 2 * (num_bath + 1), groups)

    @classmethod
    def star(
        cls,
        onsite_impurity_energy: float,
        onsite_impurity_repulsion: float,
        onsite_bath_energy: list[float],
        impurity_bath_hopping: list[float],
    ) -> Self:
        """Construct a star AIM with every bath mode coupled directly to the impurity."""
        groups = cls._star_groups(
            onsite_impurity_energy,
            onsite_impurity_repulsion,
            onsite_bath_energy,
            impurity_bath_hopping,
        )
        terms = [term for group in groups for term in group]
        return cls(
            len(onsite_bath_energy),
            onsite_impurity_energy,
            onsite_impurity_repulsion,
            "star",
            terms,
            groups,
        )

    @classmethod
    def chain(
        cls,
        onsite_impurity_energy: float,
        onsite_impurity_repulsion: float,
        onsite_bath_energy: list[float],
        bath_bath_hopping: list[float],
        impurity_bath_hopping: float,
    ) -> Self:
        """Construct a chain AIM with the impurity coupled only to the first bath site."""
        groups = cls._chain_groups(
            onsite_impurity_energy,
            onsite_impurity_repulsion,
            onsite_bath_energy,
            bath_bath_hopping,
            impurity_bath_hopping,
        )
        terms = [term for group in groups for term in group]
        return cls(
            len(onsite_bath_energy),
            onsite_impurity_energy,
            onsite_impurity_repulsion,
            "chain",
            terms,
            groups,
        )

    @staticmethod
    def _onsite_terms(
        onsite_impurity_energy: float,
        onsite_impurity_repulsion: float,
        onsite_bath_energy: list[float],
        num_bath: int,
        mode,
    ) -> list:
        """Build impurity energy, repulsion, and bath-level onsite terms."""
        onsite = []
        if onsite_impurity_energy:
            onsite += [
                -onsite_impurity_energy * number(mode(0, 0)),
                -onsite_impurity_energy * number(mode(0, 1)),
            ]
        if onsite_impurity_repulsion:
            onsite += density_density(mode(0, 0), mode(0, 1), onsite_impurity_repulsion)
        for k in range(1, num_bath + 1):
            level = onsite_bath_energy[k - 1]
            if level:
                onsite += [-level * number(mode(k, 0)), -level * number(mode(k, 1))]
        return onsite

    @staticmethod
    def _star_groups(
        onsite_impurity_energy: float,
        onsite_impurity_repulsion: float,
        onsite_bath_energy: list[float],
        impurity_bath_hopping: list[float],
    ) -> list[list]:
        """Group each impurity–bath bond separately, followed by one onsite layer."""
        if len(onsite_bath_energy) != len(impurity_bath_hopping):
            raise ValueError(
                f"onsite_bath_energy and impurity_bath_hopping must have the same length; got {len(onsite_bath_energy)} and {len(impurity_bath_hopping)}."
            )
        num_bath = len(onsite_bath_energy)

        def mode(site: int, spin: int) -> int:
            return spin * (num_bath + 1) + site

        hyb_layers = []
        for k in range(1, num_bath + 1):
            layer = hopping(
                mode(0, 0), mode(k, 0), -impurity_bath_hopping[k - 1]
            ) + hopping(mode(0, 1), mode(k, 1), -impurity_bath_hopping[k - 1])
            hyb_layers.append(layer)

        onsite = AndersonImpurity._onsite_terms(
            onsite_impurity_energy,
            onsite_impurity_repulsion,
            onsite_bath_energy,
            num_bath,
            mode,
        )
        return hyb_layers + [onsite]

    @staticmethod
    def _chain_groups(
        onsite_impurity_energy: float,
        onsite_impurity_repulsion: float,
        onsite_bath_energy: list[float],
        bath_bath_hopping: list[float],
        impurity_bath_hopping: float,
    ) -> list[list]:
        """Group chain hopping into spin-resolved even/odd layers plus one onsite layer."""
        num_bath = len(onsite_bath_energy)
        expected_t_len = max(num_bath - 1, 0)
        if len(bath_bath_hopping) != expected_t_len:
            raise ValueError(
                f"bath_bath_hopping must have length num_bath-1={expected_t_len}; got {len(bath_bath_hopping)}."
            )

        def mode(site: int, spin: int) -> int:
            return spin * (num_bath + 1) + site

        bonds = (
            LinearLattice(num_bath + 1, periodic=False).nn_pairs()
            if num_bath >= 1
            else []
        )
        couplings = [-impurity_bath_hopping, *(-t for t in bath_bath_hopping)]
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

        onsite = AndersonImpurity._onsite_terms(
            onsite_impurity_energy,
            onsite_impurity_repulsion,
            onsite_bath_energy,
            num_bath,
            mode,
        )
        return [up_even, up_odd, down_even, down_odd, onsite]
