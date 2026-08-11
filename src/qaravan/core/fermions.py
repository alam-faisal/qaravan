"""Fermion-to-qubit encoding: FermionOperator/MajoranaOperator algebra, Jordan-Wigner transform.

Physics (FermionOperator/MajoranaOperator) is encoding-agnostic; FermionicEncoding is a value
(which qubit each mode lands on, which map) threaded through both Hamiltonian and observable
construction so they can't independently drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qaravan.core.observables import PauliString, PauliSum

if TYPE_CHECKING:
    from qaravan.core.hamiltonians import Hamiltonian, HamiltonianTerm


# ---------------------------------------------------------------------------
# Ladder operators
# ---------------------------------------------------------------------------


@dataclass
class FermionTerm:
    """coeff * product of ladder operators in the given order, e.g. c-dagger_a c_b ->
    ops=[(a, True), (b, False)].

    No implicit normal-ordering: the Jordan-Wigner map is an algebra homomorphism, so
    multiplying the per-operator Pauli representations in the SAME order the ladder operators
    are written reproduces the fermionic product exactly, signs included. Order is the
    caller's responsibility. A single FermionTerm need not be individually Hermitian -- see
    jordan_wigner.
    """

    coeff: complex
    ops: list[tuple[int, bool]]

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


def c(mode: int) -> FermionTerm:
    """Annihilation operator on mode, as a length-1 FermionTerm."""
    return FermionTerm(1.0, [(mode, False)])


def cdag(mode: int) -> FermionTerm:
    """Creation operator on mode, as a length-1 FermionTerm."""
    return FermionTerm(1.0, [(mode, True)])


def number(mode: int) -> FermionTerm:
    """n_mode = c-dagger_mode c_mode -- Hermitian on its own, no partner needed at encode time."""
    return cdag(mode) @ c(mode)


def hopping(a: int, b: int, coeff: complex) -> list[FermionTerm]:
    """coeff * (c-dagger_a c_b + c-dagger_b c_a) -- two FermionTerms, neither individually
    Hermitian; jordan_wigner re-pairs them automatically at encode time."""
    return [coeff * (cdag(a) @ c(b)), coeff * (cdag(b) @ c(a))]


def density_density(a: int, b: int, coeff: complex) -> list[FermionTerm]:
    """coeff * n_a n_b -- Hermitian on its own (product of commuting self-adjoint operators).
    Single-entry list, kept as a list for uniform +='ing into a flat term list."""
    return [coeff * (number(a) @ number(b))]


@dataclass
class FermionOperator:
    """Sum of FermionTerm over a fixed number of modes. Flat -- no grouping of any kind."""

    terms: list[FermionTerm]
    num_modes: int

    def __add__(self, other: FermionOperator) -> FermionOperator:
        if self.num_modes != other.num_modes:
            raise ValueError(
                f"Cannot add FermionOperators with different num_modes: "
                f"{self.num_modes} vs {other.num_modes}."
            )
        return FermionOperator(self.terms + other.terms, self.num_modes)

    def __mul__(self, scalar: complex) -> FermionOperator:
        return FermionOperator([scalar * term for term in self.terms], self.num_modes)

    def __rmul__(self, scalar: complex) -> FermionOperator:
        return self.__mul__(scalar)

    def dagger(self) -> FermionOperator:
        return FermionOperator([term.dagger() for term in self.terms], self.num_modes)

    def encode(self, encoding: FermionicEncoding) -> Hamiltonian:
        """Apply a fermion-to-qubit encoding. The only qubit-producing method -- everything
        above is pure algebra, unaware that qubits exist."""
        from qaravan.core.hamiltonians import Hamiltonian

        if encoding.num_modes != self.num_modes:
            raise ValueError(
                f"Encoding is for {encoding.num_modes} modes; operator has {self.num_modes}."
            )
        terms = encoding.encode_fermion_terms(self.terms)
        return Hamiltonian(encoding.num_qubits, terms, None)


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


def _restrict(pauli_sum: PauliSum) -> HamiltonianTerm:
    """Embed a full-register PauliSum down to just the qubits it touches."""
    from qaravan.core.hamiltonians import HamiltonianTerm

    sites = sorted(
        {i for term in pauli_sum.terms for i, ch in enumerate(term.string) if ch != "I"}
    )
    local_terms = [
        PauliString("".join(term.string[i] for i in sites), term.coeff)
        for term in pauli_sum.terms
    ]
    return HamiltonianTerm(sites, PauliSum(local_terms))


def jordan_wigner(
    terms: list[FermionTerm],
    num_modes: int,
    mode_order: tuple[int, ...] | None = None,
) -> list[HamiltonianTerm]:
    """Map a flat list of FermionTerms to Hermitian-guaranteed HamiltonianTerms via
    Jordan-Wigner.

    mode_order[qubit] = fermionic mode occupying that qubit; defaults to the identity
    (mode i -> qubit i). Non-self-adjoint terms (e.g. one half of a hopping bond) are
    automatically paired with whichever remaining term makes their sum real --
    Hamiltonian's Trotter machinery exponentiates each returned term individually, which
    is only unitary if that term is Hermitian. Raises ValueError if a term has no such
    partner. Near-zero results are dropped rather than returned as empty HamiltonianTerms.
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
            ham_terms.append(_restrict(combined))
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


@dataclass
class MajoranaOperator:
    """Sum of MajoranaTerm over 2*num_modes Majorana operators. Mirrors FermionOperator."""

    terms: list[MajoranaTerm]
    num_modes: int

    def __add__(self, other: MajoranaOperator) -> MajoranaOperator:
        if self.num_modes != other.num_modes:
            raise ValueError(
                f"Cannot add MajoranaOperators with different num_modes: "
                f"{self.num_modes} vs {other.num_modes}."
            )
        return MajoranaOperator(self.terms + other.terms, self.num_modes)

    def __mul__(self, scalar: complex) -> MajoranaOperator:
        return MajoranaOperator([scalar * term for term in self.terms], self.num_modes)

    def __rmul__(self, scalar: complex) -> MajoranaOperator:
        return self.__mul__(scalar)


def gamma(index: int) -> MajoranaTerm:
    """Single Majorana operator gamma_index, as a length-1 MajoranaTerm."""
    return MajoranaTerm(1.0, [index])


def majorana_bilinear(a: int, b: int, coeff: complex = 1j) -> MajoranaTerm:
    """coeff * gamma_a * gamma_b -- default coeff=1j makes i*gamma_a*gamma_b Hermitian for a != b."""
    return coeff * (gamma(a) @ gamma(b))


def to_majorana(term: FermionTerm) -> MajoranaOperator:
    """Rewrite a FermionTerm in the Majorana basis: c_k = (gamma_2k + i*gamma_2k+1)/2,
    c-dagger_k = (gamma_2k - i*gamma_2k+1)/2, substituted and distributed over the product.
    An m-operator FermionTerm expands to up to 2**m MajoranaTerms -- the same combinatorial
    cost jordan_wigner already pays per term."""
    num_modes = max((mode for mode, _ in term.ops), default=-1) + 1
    expansions: list[MajoranaTerm] = [MajoranaTerm(term.coeff, [])]
    for mode, is_dagger in term.ops:
        sign = -1j if is_dagger else 1j
        expanded = []
        for partial in expansions:
            expanded.append(partial @ MajoranaTerm(0.5, [2 * mode]))
            expanded.append(partial @ MajoranaTerm(0.5 * sign, [2 * mode + 1]))
        expansions = expanded
    return MajoranaOperator(expansions, num_modes)


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
    """Encoding of num_modes fermionic modes into qubits. Threaded through both Hamiltonian
    and observable construction so they can't independently drift.
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

    def encode_fermion_terms(self, terms: list[FermionTerm]) -> list[HamiltonianTerm]:
        """Dispatches on self.name -- just jordan_wigner(...) until a second encoding exists."""
        return jordan_wigner(terms, self.num_modes, self.mode_order)

    def encode_majorana_terms(self, terms: list[MajoranaTerm]) -> PauliSum:
        return majorana_observable(terms, self.num_modes, self.mode_order)
