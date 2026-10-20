"""Fermion-to-qubit encoding: FermionicSum/MajoranaSum algebra, Jordan-Wigner transform."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from qaravan.core._repr import (
    as_math,
    elide,
    format_coefficient,
    join_terms,
    latex_ladder_ops,
    latex_majorana_ops,
    latex_term,
    math_more,
)
from qaravan.core.observables import (
    HamiltonianTerm,
    PauliString,
    PauliSum,
    restrict_pauli_sum,
)

# === Fermionic algebra ===


@dataclass
class FermionicTerm:
    """Ordered ladder-operator product with no implicit normal ordering or Hermiticity."""

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

    def normal_order(self) -> list[FermionicTerm]:
        """Expand into canonical terms with creations left and each block ascending."""
        pending = [(self.coeff, list(self.ops))]
        segregated = []
        while pending:
            coeff, ops = pending.pop()
            position = _first_annihilation_before_creation(ops)
            if position is None:
                segregated.append((coeff, ops))
                continue
            left, right = ops[position], ops[position + 1]
            pending.append(
                (-coeff, ops[:position] + [right, left] + ops[position + 2 :])
            )
            if left[0] == right[0]:
                pending.append((coeff, ops[:position] + ops[position + 2 :]))

        ordered = []
        for coeff, ops in segregated:
            sorted_ops, sign = _sort_blocks(ops)
            if sign:
                ordered.append(FermionicTerm(sign * coeff, sorted_ops))
        return ordered

    def dagger(self) -> FermionicTerm:
        """(op1...opn)† = opn†...op1†, each dagger flipped, coeff conjugated."""
        return FermionicTerm(
            np.conj(self.coeff),
            [(mode, not is_dagger) for mode, is_dagger in reversed(self.ops)],
        )

    def compose(self, other: FermionicTerm) -> FermionicTerm:
        """Concatenate operators in left-to-right product order and multiply coefficients."""
        return FermionicTerm(self.coeff * other.coeff, self.ops + other.ops)

    def __matmul__(self, other: FermionicTerm) -> FermionicTerm:
        return self.compose(other)

    def __mul__(self, scalar: complex) -> FermionicTerm:
        return FermionicTerm(self.coeff * scalar, self.ops)

    def __rmul__(self, scalar: complex) -> FermionicTerm:
        return self.__mul__(scalar)

    def to_majorana(self, num_fermionic_modes: int | None = None) -> MajoranaSum:
        """Expand into up to 2**len(ops) Majorana terms on the requested register size."""
        highest_mode = max((mode for mode, _ in self.ops), default=-1)
        if num_fermionic_modes is None:
            num_modes = highest_mode + 1
        elif num_fermionic_modes <= highest_mode:
            raise ValueError(
                f"Term acts on mode {highest_mode}, which needs at least "
                f"{highest_mode + 1} modes; got num_fermionic_modes={num_fermionic_modes}."
            )
        else:
            num_modes = num_fermionic_modes
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
        # "1" is the identity term normal_order's contractions produce
        ops_str = ops_str or "1"
        if self.coeff == 1.0:
            return ops_str
        return f"{format_coefficient(self.coeff)} {ops_str}"

    def _repr_latex_(self) -> str:
        return as_math(latex_term(self.coeff, latex_ladder_ops(self.ops)))


def c(mode: int) -> FermionicTerm:
    """Annihilation operator on mode."""
    return FermionicTerm(1.0, [(mode, False)])


def cdag(mode: int) -> FermionicTerm:
    """Creation operator on mode."""
    return FermionicTerm(1.0, [(mode, True)])


def number(mode: int) -> FermionicTerm:
    """Number operator c†_mode c_mode."""
    return cdag(mode) @ c(mode)


def hopping(mode_a: int, mode_b: int, coeff: complex) -> list[FermionicTerm]:
    """Hermitian hopping pair coeff·(c†_a c_b + c†_b c_a)."""
    return [coeff * (cdag(mode_a) @ c(mode_b)), coeff * (cdag(mode_b) @ c(mode_a))]


def density_density(mode_a: int, mode_b: int, coeff: complex) -> list[FermionicTerm]:
    """Single-term list containing coeff·n_a n_b."""
    return [coeff * (number(mode_a) @ number(mode_b))]


@dataclass
class FermionicSum:
    """Sum of FermionicTerms on a fixed fermionic register."""

    terms: list[FermionicTerm]
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
        """Return a plain canonical FermionicSum with ordered, merged terms above atol."""
        merged: dict[tuple[tuple[int, bool], ...], complex] = {}
        for term in self.terms:
            for ordered in term.normal_order():
                key = tuple(ordered.ops)
                merged[key] = merged.get(key, 0) + ordered.coeff
        terms = [
            FermionicTerm(coeff, list(ops))
            for ops, coeff in sorted(merged.items())
            if abs(coeff) > atol
        ]
        return FermionicSum(terms, self.num_fermionic_modes)

    def is_hermitian(self, atol: float = 1e-9) -> bool:
        """Compare normal-ordered forms of the operator and its adjoint."""
        canonical = {tuple(t.ops): t.coeff for t in self.normal_order().terms}
        adjoint = {tuple(t.ops): t.coeff for t in self.dagger().normal_order().terms}
        return all(
            abs(canonical.get(ops, 0) - adjoint.get(ops, 0)) < atol
            for ops in set(canonical) | set(adjoint)
        )

    def encode(self, encoding: FermionicEncoding) -> PauliSum:
        """Encode as a full-register PauliSum without requiring Hermiticity."""
        if encoding.num_modes != self.num_fermionic_modes:
            raise ValueError(
                f"Encoding is for {encoding.num_modes} modes; operator has "
                f"{self.num_fermionic_modes}."
            )
        return encoding.encode_fermionic_terms(self.terms)

    def __repr__(self) -> str:
        return join_terms(elide([repr(term) for term in self.terms]))

    def _repr_latex_(self) -> str:
        parts = [latex_term(t.coeff, latex_ladder_ops(t.ops)) for t in self.terms]
        return as_math(join_terms(elide(parts, marker=math_more)))


# === Normal ordering ===


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
    """Sort creation and annihilation blocks, returning sign zero for repeated modes."""
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


def _first_annihilation_before_creation(ops: list[tuple[int, bool]]) -> int | None:
    for position in range(len(ops) - 1):
        if not ops[position][1] and ops[position + 1][1]:
            return position
    return None


# === Majorana algebra ===


@dataclass
class MajoranaTerm:
    """Ordered product of self-adjoint Majoranas indexed in [0, 2*num_modes)."""

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
        # index 2m is gamma_m, 2m+1 is gamma-bar_m -- how the operators are written
        ops_str = " ".join(
            f"{'γ̄' if index % 2 else 'γ'}_{index // 2}" for index in self.ops
        )
        if self.coeff == 1.0:
            return ops_str
        return f"{format_coefficient(self.coeff)} {ops_str}"

    def _repr_latex_(self) -> str:
        return as_math(latex_term(self.coeff, latex_majorana_ops(self.ops)))


@dataclass
class MajoranaSum:
    """Sum of MajoranaTerms on 2*num_fermionic_modes Majorana operators."""

    terms: list[MajoranaTerm]
    num_fermionic_modes: int

    def __add__(self, other: MajoranaSum) -> MajoranaSum:
        if self.num_fermionic_modes != other.num_fermionic_modes:
            raise ValueError(
                f"Cannot add MajoranaSums with different num_fermionic_modes: "
                f"{self.num_fermionic_modes} vs {other.num_fermionic_modes}."
            )
        return MajoranaSum(self.terms + other.terms, self.num_fermionic_modes)

    def __mul__(self, scalar: complex) -> MajoranaSum:
        return MajoranaSum(
            [scalar * term for term in self.terms], self.num_fermionic_modes
        )

    def __rmul__(self, scalar: complex) -> MajoranaSum:
        return self.__mul__(scalar)

    def encode(self, encoding: FermionicEncoding) -> PauliSum:
        """Encode as a full-register PauliSum."""
        if encoding.num_modes != self.num_fermionic_modes:
            raise ValueError(
                f"Encoding is for {encoding.num_modes} modes; operator has "
                f"{self.num_fermionic_modes}."
            )
        return encoding.encode_majorana_terms(self.terms)

    def __repr__(self) -> str:
        return join_terms(elide([repr(term) for term in self.terms]))

    def _repr_latex_(self) -> str:
        parts = [latex_term(t.coeff, latex_majorana_ops(t.ops)) for t in self.terms]
        return as_math(join_terms(elide(parts, marker=math_more)))


def gamma(index: int) -> MajoranaTerm:
    """Single Majorana operator γ_index."""
    return MajoranaTerm(1.0, [index])


def majorana_bilinear(index_a: int, index_b: int, coeff: complex = 1j) -> MajoranaTerm:
    """Return coeff·γ_aγ_b; the default iγ_aγ_b is Hermitian for distinct indices."""
    return coeff * (gamma(index_a) @ gamma(index_b))


# === Jordan-Wigner transform ===

# _majorana_pauli_string backs both the Majorana and the ladder-operator paths.


def _qubit_of(num_modes: int, mode_order: tuple[int, ...] | None) -> list[int]:
    """qubit_of[mode] = the qubit that mode occupies; mode_order=None is the identity."""
    mode_order = mode_order or tuple(range(num_modes))
    table = [0] * num_modes
    for qubit, mode in enumerate(mode_order):
        table[mode] = qubit
    return table


def _majorana_pauli_string(qubit: int, is_odd: bool, num_modes: int) -> PauliString:
    """Map γ_2k to Z_<q X_q or γ_2k+1 to Z_<q Y_q."""
    chars = ["Z"] * qubit + ["Y" if is_odd else "X"] + ["I"] * (num_modes - qubit - 1)
    return PauliString("".join(chars), 1.0)


def _ladder_pauli_sum(qubit: int, is_dagger: bool, num_modes: int) -> PauliSum:
    """Map c_k=(γ_2k+iγ_2k+1)/2 or its adjoint to a PauliSum."""
    x_term = _majorana_pauli_string(
        qubit, is_odd=False, num_modes=num_modes
    )  # gamma_2k
    y_term = _majorana_pauli_string(
        qubit, is_odd=True, num_modes=num_modes
    )  # gamma_2k+1
    sign = -1j if is_dagger else 1j
    return PauliSum([0.5 * x_term, 0.5 * sign * y_term])


def _map_term(term: FermionicTerm, qubit_of: list[int], num_modes: int) -> PauliSum:
    """Jordan–Wigner-map one potentially non-Hermitian FermionicTerm."""
    result = PauliSum([PauliString("I" * num_modes, 1.0)])
    for mode, is_dagger in term.ops:
        result = result.compose(_ladder_pauli_sum(qubit_of[mode], is_dagger, num_modes))
    return term.coeff * result


def _is_real(pauli_sum: PauliSum, atol: float = 1e-9) -> bool:
    return all(abs(term.coeff.imag) < atol for term in pauli_sum.terms)


def jordan_wigner(
    terms: list[FermionicTerm],
    num_modes: int,
    mode_order: tuple[int, ...] | None = None,
) -> PauliSum:
    """Map terms independently to a full-register PauliSum without requiring Hermiticity."""
    qubit_of = _qubit_of(num_modes, mode_order)
    mapped = []
    for term in terms:
        mapped.extend(_map_term(term, qubit_of, num_modes).terms)
    return PauliSum(mapped).simplify()


def jordan_wigner_hamiltonian(
    terms: list[FermionicTerm],
    num_modes: int,
    mode_order: tuple[int, ...] | None = None,
) -> list[HamiltonianTerm]:
    """Jordan–Wigner-map terms, pairing adjoints into site-restricted Hermitian terms."""
    qubit_of = _qubit_of(num_modes, mode_order)

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
                    f"FermionicTerm {terms[i]} has no Hermitian-conjugate partner among "
                    "the remaining terms. The operator may still be Hermitian overall -- "
                    "normal_order() merges monomials split across several terms."
                )
            combined = (image + images[partner]).simplify()
            used[partner] = True
        used[i] = True
        if any(abs(term.coeff) > 1e-12 for term in combined.terms):
            ham_terms.append(restrict_pauli_sum(combined))
    return ham_terms


def jordan_wigner_majorana(
    terms: list[MajoranaTerm],
    num_modes: int,
    mode_order: tuple[int, ...] | None = None,
) -> PauliSum:
    """Map MajoranaTerms one-to-one onto full-register PauliStrings."""
    qubit_of = _qubit_of(num_modes, mode_order)

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


@dataclass(frozen=True)
class FermionicEncoding:
    """Fermion-to-qubit encoding with mode_order[qubit] giving the resident mode."""

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
        """Qubits holding spin-up [0,M) and spin-down [M,2M) modes."""
        if self.num_modes % 2:
            raise ValueError(
                f"Spinful mode labelling needs an even mode count; got {self.num_modes}."
            )
        half = self.num_modes // 2
        up = [self.qubit_of(mode) for mode in range(half)]
        down = [self.qubit_of(mode) for mode in range(half, self.num_modes)]
        return up, down

    def encode_hamiltonian_terms(
        self, terms: list[FermionicTerm]
    ) -> list[HamiltonianTerm]:
        """Encode Hamiltonian terms with Jordan–Wigner."""
        return jordan_wigner_hamiltonian(terms, self.num_modes, self.mode_order)

    def encode_fermionic_terms(self, terms: list[FermionicTerm]) -> PauliSum:
        """Encode fermionic terms with Jordan–Wigner."""
        return jordan_wigner(terms, self.num_modes, self.mode_order)

    def encode_majorana_terms(self, terms: list[MajoranaTerm]) -> PauliSum:
        """Encode Majorana terms with Jordan–Wigner."""
        return jordan_wigner_majorana(terms, self.num_modes, self.mode_order)
