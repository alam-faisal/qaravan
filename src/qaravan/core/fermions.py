"""Fermion-to-qubit encoding: FermionicSum/MajoranaSum algebra, Jordan-Wigner transform."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from qaravan.core.observables import (
    HamiltonianTerm,
    PauliString,
    PauliSum,
    embed_hamiltonian_terms,
    restrict_pauli_sum,
)

# ---------------------------------------------------------------------------
# Ladder operators
# ---------------------------------------------------------------------------


@dataclass
class FermionTerm:
    """coeff * product of ladder operators in the given order, e.g. c-dagger_a c_b ->
    ops=[(a, True), (b, False)].

    No implicit normal-ordering -- order is the caller's responsibility and is preserved
    exactly through Jordan-Wigner, so two terms denoting the same operator can hold different
    ops (n_a n_b vs n_b n_a). Compare them via FermionicSum.normal_order(). A single FermionTerm
    need not be individually Hermitian
    """

    coeff: complex
    ops: list[tuple[int, bool]]

    @property
    def normal_ordered(self) -> bool:
        """All creations, then all annihilations, each block strictly ascending in mode."""
        daggers = [is_dagger for _, is_dagger in self.ops]
        if daggers != sorted(daggers, reverse=True):
            return False
        split = sum(daggers)
        creations = [mode for mode, _ in self.ops[:split]]
        annihilations = [mode for mode, _ in self.ops[split:]]
        return _strictly_ascending(creations) and _strictly_ascending(annihilations)

    def dagger(self) -> FermionTerm:
        """(op1...opn)† = opn†...op1†, each dagger flipped, coeff conjugated."""
        return FermionTerm(
            np.conj(self.coeff),
            [(mode, not is_dagger) for mode, is_dagger in reversed(self.ops)],
        )

    def compose(self, other: FermionTerm) -> FermionTerm:
        """Concatenate ops, multiply coeffs. Order preserved: self's ops come first."""
        return FermionTerm(self.coeff * other.coeff, self.ops + other.ops)

    def __matmul__(self, other: FermionTerm) -> FermionTerm:
        return self.compose(other)

    def __mul__(self, scalar: complex) -> FermionTerm:
        return FermionTerm(self.coeff * scalar, self.ops)

    def __rmul__(self, scalar: complex) -> FermionTerm:
        return self.__mul__(scalar)

    def to_majorana(self) -> MajoranaSum:
        """Rewrite in the Majorana basis (c_k = (gamma_2k + i*gamma_2k+1)/2,
        c-dagger_k = (gamma_2k - i*gamma_2k+1)/2), substituted and distributed over the
        product. Expands to up to 2**len(self.ops) MajoranaTerms."""
        num_modes = max((mode for mode, _ in self.ops), default=-1) + 1
        expansions = [MajoranaTerm(self.coeff, [])]
        for mode, is_dagger in self.ops:
            sign = -1j if is_dagger else 1j
            expanded = []
            for partial in expansions:
                expanded.append(partial @ MajoranaTerm(0.5, [2 * mode]))
                expanded.append(partial @ MajoranaTerm(0.5 * sign, [2 * mode + 1]))
            expansions = expanded
        return MajoranaSum(expansions, num_modes)

    def __repr__(self) -> str:
        ops_str = " ".join(
            f"c†_{mode}" if is_dagger else f"c_{mode}" for mode, is_dagger in self.ops
        )
        if self.coeff == 1.0:
            return ops_str
        return f"{self.coeff} * {ops_str}"


def c(mode: int) -> FermionTerm:
    """Annihilation operator on mode, as a length-1 FermionTerm."""
    return FermionTerm(1.0, [(mode, False)])


def cdag(mode: int) -> FermionTerm:
    """Creation operator on mode, as a length-1 FermionTerm."""
    return FermionTerm(1.0, [(mode, True)])


def number(mode: int) -> FermionTerm:
    """n_mode = c-dagger_mode c_mode -- Hermitian on its own, no partner needed at encode time."""
    return cdag(mode) @ c(mode)


def hopping(mode_a: int, mode_b: int, coeff: complex) -> list[FermionTerm]:
    """coeff * (c-dagger_a c_b + c-dagger_b c_a) -- two FermionTerms, neither individually
    Hermitian; jordan_wigner re-pairs them automatically at encode time."""
    return [coeff * (cdag(mode_a) @ c(mode_b)), coeff * (cdag(mode_b) @ c(mode_a))]


def density_density(mode_a: int, mode_b: int, coeff: complex) -> list[FermionTerm]:
    """coeff * n_a n_b -- Hermitian on its own (product of commuting self-adjoint operators).
    Single-entry list, kept as a list for uniform +='ing into a flat term list."""
    return [coeff * (number(mode_a) @ number(mode_b))]


# ---------------------------------------------------------------------------
# Normal ordering
# ---------------------------------------------------------------------------


def _strictly_ascending(modes: list[int]) -> bool:
    return all(earlier < later for earlier, later in zip(modes, modes[1:]))


def _sort_sign(modes: list[int]) -> int:
    """(-1)^inversions: sign of the permutation sorting distinct modes ascending."""
    inversions = 0
    for position, earlier in enumerate(modes):
        for later in modes[position + 1 :]:
            if earlier > later:
                inversions += 1
    return -1 if inversions % 2 else 1


def _sort_blocks(ops: list[tuple[int, bool]]) -> tuple[list[tuple[int, bool]], int]:
    """Sort each block of an already dagger-segregated ops list; sign 0 if a mode repeats,
    since c†_i c†_i = c_i c_i = 0."""
    split = sum(1 for _, is_dagger in ops if is_dagger)
    creations = [mode for mode, _ in ops[:split]]
    annihilations = [mode for mode, _ in ops[split:]]
    if len(set(creations)) != len(creations) or len(set(annihilations)) != len(
        annihilations
    ):
        return [], 0
    sign = _sort_sign(creations) * _sort_sign(annihilations)
    sorted_ops = [(mode, True) for mode in sorted(creations)]
    sorted_ops += [(mode, False) for mode in sorted(annihilations)]
    return sorted_ops, sign


def _normal_order_term(term: FermionTerm) -> list[FermionTerm]:
    """One term as a sum of normal-ordered terms -- a sum, not a term, because the
    c_i c†_j = delta_ij - c†_j c_i rewrite spawns a shorter contracted term whenever i == j."""
    pending = [(term.coeff, list(term.ops))]
    segregated = []
    while pending:
        coeff, ops = pending.pop()
        position = _first_annihilation_before_creation(ops)
        if position is None:
            segregated.append((coeff, ops))
            continue
        left, right = ops[position], ops[position + 1]
        pending.append((-coeff, ops[:position] + [right, left] + ops[position + 2 :]))
        if left[0] == right[0]:
            pending.append((coeff, ops[:position] + ops[position + 2 :]))

    ordered = []
    for coeff, ops in segregated:
        sorted_ops, sign = _sort_blocks(ops)
        if sign:
            ordered.append(FermionTerm(sign * coeff, sorted_ops))
    return ordered


def _first_annihilation_before_creation(ops: list[tuple[int, bool]]) -> int | None:
    for position in range(len(ops) - 1):
        if not ops[position][1] and ops[position + 1][1]:
            return position
    return None


@dataclass
class FermionicSum:
    """Sum of FermionTerm"""

    terms: list[FermionTerm]
    num_fermionic_modes: int

    def __add__(self, other: FermionicSum) -> FermionicSum:
        if self.num_fermionic_modes != other.num_fermionic_modes:
            raise ValueError(
                f"Cannot add FermionicSums with different num_fermionic_modes: "
                f"{self.num_fermionic_modes} vs {other.num_fermionic_modes}."
            )
        return FermionicSum(self.terms + other.terms, self.num_fermionic_modes)

    def __mul__(self, scalar: complex) -> FermionicSum:
        return FermionicSum(
            [scalar * term for term in self.terms], self.num_fermionic_modes
        )

    def __rmul__(self, scalar: complex) -> FermionicSum:
        return self.__mul__(scalar)

    def dagger(self) -> FermionicSum:
        return FermionicSum(
            [term.dagger() for term in self.terms], self.num_fermionic_modes
        )

    def normal_order(self, atol: float = 1e-12) -> FermionicSum:
        """Canonical equivalent: every term normal_ordered, equal monomials merged, coefficients
        below atol dropped, terms sorted -- so equal operators give equal term lists.

        Returns a plain FermionicSum even from a subclass: the canonical form is a different
        term decomposition, so a FermionicHamiltonian's trotter_groups would not apply to it.
        """
        merged: dict[tuple[tuple[int, bool], ...], complex] = {}
        for term in self.terms:
            for ordered in _normal_order_term(term):
                key = tuple(ordered.ops)
                merged[key] = merged.get(key, 0) + ordered.coeff
        terms = [
            FermionTerm(coeff, list(ops))
            for ops, coeff in sorted(merged.items())
            if abs(coeff) > atol
        ]
        return FermionicSum(terms, self.num_fermionic_modes)

    def is_hermitian(self, atol: float = 1e-9) -> bool:
        """H is Hermitian iff H and H.dagger() share a normal-ordered form -- purely fermionic,
        no encoding involved."""
        canonical = {tuple(t.ops): t.coeff for t in self.normal_order().terms}
        adjoint = {tuple(t.ops): t.coeff for t in self.dagger().normal_order().terms}
        return all(
            abs(canonical.get(ops, 0) - adjoint.get(ops, 0)) < atol
            for ops in set(canonical) | set(adjoint)
        )

    def __repr__(self) -> str:
        return " + ".join(repr(term) for term in self.terms)

    def encode(self, encoding: FermionicEncoding) -> PauliSum:
        """Apply a fermion-to-qubit encoding, converting the abstract fermionic algebra
        into concrete qubit operators."""
        if encoding.num_modes != self.num_fermionic_modes:
            raise ValueError(
                f"Encoding is for {encoding.num_modes} modes; operator has "
                f"{self.num_fermionic_modes}."
            )
        terms = encoding.encode_fermion_terms(self.terms)
        return embed_hamiltonian_terms(terms, encoding.num_qubits)


# ---------------------------------------------------------------------------
# Shared Jordan-Wigner primitive (also backs Majorana operators, below)
# ---------------------------------------------------------------------------


def _majorana_pauli_string(qubit: int, is_odd: bool, num_modes: int) -> PauliString:
    """gamma_2k -> Z_{<q} X_q ; gamma_2k+1 -> Z_{<q} Y_q (q = qubit_of[mode]). Single string,
    coeff 1 -- no PauliSum needed for one Majorana operator."""
    chars = ["Z"] * qubit + ["Y" if is_odd else "X"] + ["I"] * (num_modes - qubit - 1)
    return PauliString("".join(chars), 1.0)


def _ladder_pauli_sum(qubit: int, is_dagger: bool, num_modes: int) -> PauliSum:
    """c_k = (gamma_2k + i*gamma_2k+1)/2 ; c-dagger_k = (gamma_2k - i*gamma_2k+1)/2."""
    x_term = _majorana_pauli_string(
        qubit, is_odd=False, num_modes=num_modes
    )  # gamma_2k
    y_term = _majorana_pauli_string(
        qubit, is_odd=True, num_modes=num_modes
    )  # gamma_2k+1
    sign = -1j if is_dagger else 1j
    return PauliSum([0.5 * x_term, 0.5 * sign * y_term])


# ---------------------------------------------------------------------------
# Jordan-Wigner transform (fermionic ladder operators -> HamiltonianTerm)
# ---------------------------------------------------------------------------


def _map_term(term: FermionTerm, qubit_of: list[int], num_modes: int) -> PauliSum:
    """JW-map a single FermionTerm to a PauliSum (not necessarily Hermitian on its own)."""
    result = PauliSum([PauliString("I" * num_modes, 1.0)])
    for mode, is_dagger in term.ops:
        result = result.compose(_ladder_pauli_sum(qubit_of[mode], is_dagger, num_modes))
    return term.coeff * result


def _is_real(pauli_sum: PauliSum, atol: float = 1e-9) -> bool:
    return all(abs(term.coeff.imag) < atol for term in pauli_sum.terms)


def jordan_wigner(
    terms: list[FermionTerm],
    num_modes: int,
    mode_order: tuple[int, ...] | None = None,
) -> list[HamiltonianTerm]:
    """Map a flat list of FermionTerms to Hermitian-guaranteed HamiltonianTerms via
    Jordan-Wigner (each returned term must be Hermitian so it can later be exponentiated
    into a unitary Trotter layer).

    mode_order[qubit] = fermionic mode occupying that qubit; defaults to the identity
    (mode i -> qubit i). Non-self-adjoint terms (e.g. one half of a hopping bond) are
    automatically paired with whichever remaining term makes their sum real; raises
    ValueError if no such partner exists. Near-zero results are dropped rather than
    returned as empty HamiltonianTerms.
    """
    mode_order = mode_order or tuple(range(num_modes))
    qubit_of = [0] * num_modes
    for qubit, mode in enumerate(mode_order):
        qubit_of[mode] = qubit

    images = [_map_term(term, qubit_of, num_modes) for term in terms]
    used = [False] * len(terms)
    ham_terms = []
    for i, image in enumerate(images):
        if used[i]:
            continue
        combined = image.simplify()
        if not _is_real(combined):
            partner = next(
                (
                    j
                    for j in range(i + 1, len(terms))
                    if not used[j] and _is_real((image + images[j]).simplify())
                ),
                None,
            )
            if partner is None:
                raise ValueError(
                    f"FermionTerm {terms[i]} has no Hermitian-conjugate partner among the "
                    "remaining terms; the operator, as given, is not Hermitian."
                )
            combined = (image + images[partner]).simplify()
            used[partner] = True
        used[i] = True
        if any(abs(term.coeff) > 1e-12 for term in combined.terms):
            ham_terms.append(restrict_pauli_sum(combined))
    return ham_terms


# ---------------------------------------------------------------------------
# Majorana operators
# ---------------------------------------------------------------------------


@dataclass
class MajoranaTerm:
    """coeff * product of Majorana operators. ops are plain indices in [0, 2*num_modes) --
    no dagger flag, since every gamma is self-adjoint. Order preserved (gammas anticommute)."""

    coeff: complex
    ops: list[int]

    def dagger(self) -> MajoranaTerm:
        """(g1...gn)† = gn...g1 (each gamma self-adjoint), coeff conjugated."""
        return MajoranaTerm(np.conj(self.coeff), list(reversed(self.ops)))

    def compose(self, other: MajoranaTerm) -> MajoranaTerm:
        return MajoranaTerm(self.coeff * other.coeff, self.ops + other.ops)

    def __matmul__(self, other: MajoranaTerm) -> MajoranaTerm:
        return self.compose(other)

    def __mul__(self, scalar: complex) -> MajoranaTerm:
        return MajoranaTerm(self.coeff * scalar, self.ops)

    def __rmul__(self, scalar: complex) -> MajoranaTerm:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        ops_str = " ".join(f"γ_{i}" for i in self.ops)
        if self.coeff == 1.0:
            return ops_str
        return f"{self.coeff} * {ops_str}"


@dataclass
class MajoranaSum:
    """Sum of MajoranaTerm over 2*num_modes Majorana operators. Mirrors FermionicSum."""

    terms: list[MajoranaTerm]
    num_modes: int

    def __add__(self, other: MajoranaSum) -> MajoranaSum:
        if self.num_modes != other.num_modes:
            raise ValueError(
                f"Cannot add MajoranaSums with different num_modes: "
                f"{self.num_modes} vs {other.num_modes}."
            )
        return MajoranaSum(self.terms + other.terms, self.num_modes)

    def __mul__(self, scalar: complex) -> MajoranaSum:
        return MajoranaSum([scalar * term for term in self.terms], self.num_modes)

    def __rmul__(self, scalar: complex) -> MajoranaSum:
        return self.__mul__(scalar)

    def __repr__(self) -> str:
        return " + ".join(repr(term) for term in self.terms)


def gamma(index: int) -> MajoranaTerm:
    """Single Majorana operator gamma_index, as a length-1 MajoranaTerm."""
    return MajoranaTerm(1.0, [index])


def majorana_bilinear(index_a: int, index_b: int, coeff: complex = 1j) -> MajoranaTerm:
    """coeff * gamma_a * gamma_b -- default coeff=1j makes i*gamma_a*gamma_b Hermitian for a != b."""
    return coeff * (gamma(index_a) @ gamma(index_b))


def majorana_observable(
    terms: list[MajoranaTerm],
    num_modes: int,
    mode_order: tuple[int, ...] | None = None,
) -> PauliSum:
    """JW-map and sum a list of MajoranaTerms into one full-register PauliSum observable.
    Each term maps to exactly one PauliString, so this costs len(terms) compositions total --
    no combinatorial fan-out, no site-restriction (observables want the full register, unlike
    HamiltonianTerm). Ready to pass straight to state.expectation()."""
    mode_order = mode_order or tuple(range(num_modes))
    qubit_of = [0] * num_modes
    for qubit, mode in enumerate(mode_order):
        qubit_of[mode] = qubit

    mapped = []
    for term in terms:
        image = PauliString("I" * num_modes, term.coeff)
        for index in term.ops:
            mode, is_odd = index // 2, index % 2 == 1
            image = image.compose(
                _majorana_pauli_string(qubit_of[mode], is_odd, num_modes)
            )
        mapped.append(image)
    return PauliSum(mapped).simplify()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FermionicEncoding:
    """Encoding of num_modes fermionic modes into qubits. Pass the same instance to both
    Hamiltonian and observable construction so qubit assignments stay consistent.
    mode_order[qubit] = mode occupying that qubit; None = identity."""

    name: str
    num_modes: int
    mode_order: tuple[int, ...] | None = None

    @property
    def num_qubits(self) -> int:
        return self.num_modes  # 1:1 for Jordan-Wigner; a future encoding might differ

    def qubit_of(self, mode: int) -> int:
        """Inverse of mode_order; identity if mode_order is None."""
        if self.mode_order is None:
            return mode
        return self.mode_order.index(mode)

    def spin_qubits(self) -> tuple[list[int], list[int]]:
        """Qubits holding the up and the down modes, under the spinful mode labelling
        [0, M) up and [M, 2M) down. Meaningless for a spinless operator."""
        if self.num_modes % 2:
            raise ValueError(
                f"Spinful mode labelling needs an even mode count; got {self.num_modes}."
            )
        half = self.num_modes // 2
        up = [self.qubit_of(mode) for mode in range(half)]
        down = [self.qubit_of(mode) for mode in range(half, self.num_modes)]
        return up, down

    def encode_fermion_terms(self, terms: list[FermionTerm]) -> list[HamiltonianTerm]:
        """Currently always applies Jordan-Wigner, regardless of self.name."""
        return jordan_wigner(terms, self.num_modes, self.mode_order)

    def encode_majorana_terms(self, terms: list[MajoranaTerm]) -> PauliSum:
        """Currently always applies Jordan-Wigner, regardless of self.name."""
        return majorana_observable(terms, self.num_modes, self.mode_order)


def interleaved_mode_order(
    num_fermionic_sites: int, annotate: bool = False
) -> tuple[int, ...] | tuple[tuple[int, ...], str]:
    """mode_order for FermiHubbard1D's interleaved layout: qubit 2j <- up_j, qubit 2j+1 <- down_j.

    annotate=True also returns a human-readable qubit-layout string, e.g.
    'q0=↑0 q1=↓0 q2=↑1 q3=↓1', as (mode_order, annotation).
    """
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
