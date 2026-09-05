"""Tests for core/fermions.py: FermionicSum algebra, Jordan-Wigner, Majorana operators."""

import numpy as np
import pytest
from qaravan.backends.statevector import Statevector
from qaravan.core.fermions import (
    FermionicEncoding,
    FermionicSum,
    FermionTerm,
    _is_real,
    _map_term,
    c,
    cdag,
    density_density,
    gamma,
    hopping,
    jordan_wigner,
    majorana_bilinear,
    majorana_observable,
    number,
)
from qaravan.core.observables import PauliString, PauliSum

# ---------------------------------------------------------------------------
# Ladder operators -> Jordan-Wigner (the two building blocks every fermionic
# model reduces to: an on-site number operator and a hopping bond)
# ---------------------------------------------------------------------------


def test_number_maps_to_occupation_projector():
    terms = jordan_wigner([number(0)], 1)
    assert len(terms) == 1
    expected = 0.5 * np.eye(2) - 0.5 * PauliString("Z").matrix
    assert np.allclose(terms[0].local_obs.matrix, expected)
    assert terms[0].sites == [0]


def test_hopping_bond_maps_to_weight_two_xx_plus_yy():
    # doc eq. 15: c-dagger_a c_b + h.c. -> (1/2)(XX + YY); hopping's coeff scales the whole
    # bilinear, matching the physical Hamiltonian's -t prefactor directly.
    t = 1.3
    terms = jordan_wigner(hopping(0, 1, -t), 2)
    assert len(terms) == 1
    expected = (-t / 2) * (PauliString("XX").matrix + PauliString("YY").matrix)
    assert np.allclose(terms[0].local_obs.matrix, expected)
    assert terms[0].sites == [0, 1]


def test_hopping_with_spectator_gives_weight_three_form():
    # doc eq. 6/8: one spectator mode in between leaves a residual Z (weight-3 XZX + YZY).
    t = 0.7
    terms = jordan_wigner(hopping(0, 2, -t), 3)
    assert len(terms) == 1
    expected = (-t / 2) * (PauliString("XZX").matrix + PauliString("YZY").matrix)
    assert np.allclose(terms[0].local_obs.matrix, expected)
    assert terms[0].sites == [0, 1, 2]


def test_density_density_maps_to_diagonal_form():
    U = 2.0
    terms = jordan_wigner(density_density(0, 1, U), 2)
    n = 0.5 * np.eye(2) - 0.5 * PauliString("Z").matrix
    expected = U * np.kron(n, n)
    assert np.allclose(terms[0].local_obs.matrix, expected)


def test_jordan_wigner_respects_mode_order():
    # Swapping which qubit a mode lands on should permute which qubits the term touches.
    terms_identity = jordan_wigner(hopping(0, 1, -1.0), 2, mode_order=(0, 1))
    terms_swapped = jordan_wigner(hopping(0, 1, -1.0), 2, mode_order=(1, 0))
    assert terms_identity[0].sites == [0, 1]
    assert terms_swapped[0].sites == [0, 1]
    assert np.allclose(
        terms_identity[0].local_obs.matrix, terms_swapped[0].local_obs.matrix
    )


def test_jordan_wigner_drops_zero_terms():
    # mu=0 -> the term is identically zero and shouldn't appear as a degenerate HamiltonianTerm.
    terms = jordan_wigner([0.0 * number(0)], 1)
    assert terms == []


def test_jordan_wigner_raises_on_unpaired_non_hermitian_term():
    with pytest.raises(ValueError, match="Hermitian-conjugate partner"):
        jordan_wigner([c(0)], 1)


# ---------------------------------------------------------------------------
# FermionicSum
# ---------------------------------------------------------------------------


def test_fermion_operator_add_and_scale():
    op1 = FermionicSum([number(0)], 2)
    op2 = FermionicSum(hopping(0, 1, -1.0), 2)
    combined = op1 + 2 * op2
    assert len(combined.terms) == 3
    assert combined.num_fermionic_modes == 2


def test_fermion_operator_add_mismatched_num_modes_raises():
    op1 = FermionicSum([number(0)], 2)
    op2 = FermionicSum([number(0)], 3)
    with pytest.raises(ValueError):
        op1 + op2


def test_fermion_operator_dagger_of_hopping_is_itself():
    op = FermionicSum(hopping(0, 1, -1.0), 2)
    dagger_matrix_sum = sum(
        jw.local_obs.matrix for jw in jordan_wigner(op.dagger().terms, 2)
    )
    original_matrix_sum = sum(jw.local_obs.matrix for jw in jordan_wigner(op.terms, 2))
    assert np.allclose(dagger_matrix_sum, original_matrix_sum)


def test_fermion_operator_encode_raises_on_mode_mismatch():
    op = FermionicSum([number(0)], 2)
    encoding = FermionicEncoding("jordan_wigner", num_modes=3)
    with pytest.raises(ValueError, match="modes"):
        op.encode(encoding)


def test_fermion_operator_encode_returns_pauli_sum_with_correct_qubits():
    """FermionicSum has no trotter_groups concept, so .encode() returns a plain PauliSum,
    not a Hamiltonian -- FermionicHamiltonian overrides encode() to return a Hamiltonian
    instead, since it's the one that carries trotter_groups."""
    op = FermionicSum(hopping(0, 1, -1.0), 2)
    encoding = FermionicEncoding("jordan_wigner", num_modes=2)
    obs = op.encode(encoding)
    assert isinstance(obs, PauliSum)
    assert obs.num_sites == 2
    assert obs.matrix.shape == (4, 4)


# ---------------------------------------------------------------------------
# FermionicEncoding
# ---------------------------------------------------------------------------


def test_fermionic_encoding_equality():
    a = FermionicEncoding("jordan_wigner", num_modes=4, mode_order=(1, 0, 3, 2))
    b = FermionicEncoding("jordan_wigner", num_modes=4, mode_order=(1, 0, 3, 2))
    assert a == b
    assert hash(a) == hash(b)


def test_fermionic_encoding_qubit_of_identity():
    encoding = FermionicEncoding("jordan_wigner", num_modes=3)
    assert [encoding.qubit_of(m) for m in range(3)] == [0, 1, 2]


def test_fermionic_encoding_qubit_of_permuted():
    encoding = FermionicEncoding("jordan_wigner", num_modes=3, mode_order=(2, 0, 1))
    # qubit 0 holds mode 2, qubit 1 holds mode 0, qubit 2 holds mode 1
    assert encoding.qubit_of(2) == 0
    assert encoding.qubit_of(0) == 1
    assert encoding.qubit_of(1) == 2


# ---------------------------------------------------------------------------
# Majorana operators
# ---------------------------------------------------------------------------


def test_gamma_squared_is_identity():
    squared = gamma(0).compose(gamma(0))
    assert squared.ops == [0, 0]
    assert squared.coeff == 1.0


def test_majorana_bilinear_matches_fermionic_identity():
    # i*gamma_0*gamma_1 on one mode (indices 0,1 both belong to mode 0); cross-check against
    # the c/c-dagger definition directly: gamma_0 = c+cdag, gamma_1 = i(cdag-c), so
    # i*gamma_0*gamma_1 = i*(c+cdag)*i*(cdag-c) = -(c+cdag)(cdag-c)
    #                   = -(c cdag - c c + cdag cdag - cdag c) = -(c cdag - cdag c) = -Z
    # (using c c = cdag cdag = 0, c cdag + cdag c = 1 => c cdag - cdag c = 1 - 2 cdag c = Z)
    obs = majorana_observable([majorana_bilinear(0, 1)], 1)
    assert np.allclose(obs.matrix, -PauliString("Z").matrix)


def test_majorana_observable_two_mode_bilinear():
    obs = majorana_observable([majorana_bilinear(0, 3)], 2)
    assert obs.matrix.shape == (4, 4)
    assert np.allclose(
        obs.matrix, obs.matrix.conj().T
    )  # Hermitian by construction (coeff=1j)


def test_majorana_observable_expectation_on_statevector():
    obs = majorana_observable([majorana_bilinear(0, 1)], 1)  # == -Z, per identity above
    up = Statevector(bitstring="0")
    down = Statevector(bitstring="1")
    assert np.isclose(up.expectation(obs), -1.0)
    assert np.isclose(down.expectation(obs), 1.0)


def test_to_majorana_round_trips_through_jordan_wigner():
    # number(0) mapped via fermion JW should match number(0) mapped via majorana_observable
    # after to_majorana -- two independent routes to the same operator.
    direct = jordan_wigner([number(0)], 1)[0].local_obs.matrix
    via_majorana = majorana_observable(number(0).to_majorana().terms, 1).matrix
    assert np.allclose(direct, via_majorana)


def test_to_majorana_on_hopping_term_matches_direct_jw():
    term = hopping(0, 1, -1.0)[
        0
    ]  # cdag(0) c(1) scaled -- not Hermitian alone, that's fine here
    direct = jordan_wigner([term, term.dagger()], 2)[0].local_obs.matrix
    maj = term.to_majorana()
    maj_dagger = term.dagger().to_majorana()
    combined = maj + maj_dagger
    via_majorana = majorana_observable(combined.terms, 2).matrix
    assert np.allclose(direct, via_majorana)


# ---------------------------------------------------------------------------
# normal_order and is_hermitian
#
# is_hermitian used to route through Jordan-Wigner. That implementation is kept below as
# _is_hermitian_via_jw -- an independent oracle, since a false positive (claiming Hermitian
# when not) is silent, while a false negative fails loudly at FermionicHamiltonian construction.
# ---------------------------------------------------------------------------


def _is_hermitian_via_jw(op: FermionicSum, atol: float = 1e-9) -> bool:
    """Oracle: JW-map every term and check the Pauli coefficients are real."""
    qubit_of = list(range(op.num_fermionic_modes))
    flat_terms = []
    for term in op.terms:
        flat_terms.extend(_map_term(term, qubit_of, op.num_fermionic_modes).terms)
    return _is_real(PauliSum(flat_terms).simplify(), atol)


def _canonical(op: FermionicSum) -> dict:
    return {tuple(t.ops): t.coeff for t in op.normal_order().terms}


def test_normal_order_merges_reordered_density_density():
    # n_a n_b and n_b n_a are the same operator with different ops -- the collision that made
    # a raw (ops, coeff) comparison against dagger() wrong
    forward = FermionicSum([number(0) @ number(1)], 2)
    backward = FermionicSum([number(1) @ number(0)], 2)
    assert _canonical(forward) == _canonical(backward)


def test_normal_order_kills_repeated_mode():
    assert FermionicSum([cdag(0) @ cdag(0)], 2).normal_order().terms == []
    assert FermionicSum([c(0) @ c(0)], 2).normal_order().terms == []


def test_normal_order_anticommutes_creations():
    forward = _canonical(FermionicSum([cdag(0) @ cdag(1)], 2))
    backward = _canonical(FermionicSum([cdag(1) @ cdag(0)], 2))
    assert forward == {key: -value for key, value in backward.items()}


def test_normal_order_contraction_emits_identity():
    # c_0 c†_0 = 1 - c†_0 c_0
    canonical = _canonical(FermionicSum([c(0) @ cdag(0)], 1))
    assert canonical == {(): 1.0, ((0, True), (0, False)): -1.0}


def test_normal_order_output_is_normal_ordered_and_idempotent():
    op = FermionicSum([number(1) @ number(0), *hopping(0, 2, -1.0), c(2) @ cdag(1)], 3)
    once = op.normal_order()
    assert all(term.normal_ordered for term in once.terms)
    assert _canonical(once) == _canonical(op)


def _matrix_via_jw(op: FermionicSum) -> np.ndarray:
    """Raw JW matrix. Uses _map_term rather than encode(), which routes through
    jordan_wigner and refuses operators whose terms have no Hermitian-conjugate partner."""
    dim = 2**op.num_fermionic_modes
    if not op.terms:
        return np.zeros((dim, dim), dtype=complex)
    qubit_of = list(range(op.num_fermionic_modes))
    flat = []
    for term in op.terms:
        flat.extend(_map_term(term, qubit_of, op.num_fermionic_modes).terms)
    return PauliSum(flat).simplify().matrix


@pytest.mark.parametrize("num_modes", [2, 3, 4])
def test_normal_order_preserves_the_operator(num_modes):
    """Strongest check: canonicalizing must not change the operator it denotes. Validated
    against the JW path, which is tested independently above."""
    rng = np.random.default_rng(num_modes)
    terms = []
    for _ in range(6):
        length = int(rng.integers(1, 5))
        ops = [
            (int(rng.integers(0, num_modes)), bool(rng.integers(0, 2)))
            for _ in range(length)
        ]
        terms.append(FermionTerm(complex(rng.normal(), rng.normal()), ops))

    op = FermionicSum(terms, num_modes)
    np.testing.assert_allclose(
        _matrix_via_jw(op.normal_order()), _matrix_via_jw(op), atol=1e-10
    )


def test_is_hermitian_matches_the_jw_oracle_on_real_models():
    from qaravan.core.hamiltonians import AndersonImpurity, FermiHubbard1D

    models = [FermiHubbard1D(L, t=1.0, U=2.0, mu=0.3) for L in [2, 3]]
    models.append(FermiHubbard1D(3, t=1.0, U=2.0, mu=0.3, periodic=True))
    models.append(
        AndersonImpurity.star(eps_d=-1.0, U=2.0, eps_bath=[0.5, -0.5], V=[0.3, 0.4])
    )
    models.append(
        AndersonImpurity.chain(
            eps_d=-1.0, U=2.0, eps_chain=[0.5, -0.5], t_chain=[0.2], V=0.3
        )
    )
    for model in models:
        plain = FermionicSum(model.terms, model.num_fermionic_modes)
        assert plain.is_hermitian() is True
        assert _is_hermitian_via_jw(plain) is True


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda: [number(0)], True),
        (lambda: hopping(0, 1, -1.0), True),
        (lambda: density_density(0, 1, 2.0), True),
        (lambda: [c(0)], False),
        (lambda: [cdag(0)], False),
        (lambda: [cdag(0) @ c(1)], False),  # missing its h.c. partner
        (lambda: [1j * number(0)], False),
        (lambda: [number(0) @ number(1), number(1) @ number(0)], True),
    ],
)
def test_is_hermitian_matches_the_jw_oracle(build, expected):
    op = FermionicSum(build(), 2)
    assert op.is_hermitian() is expected
    assert _is_hermitian_via_jw(op) is expected


@pytest.mark.parametrize(
    "second,expected",
    [
        ([c(1) @ c(0)], True),  # (c†_0 c†_1)† = c_1 c_0
        ([c(0) @ c(1)], False),  # differs by the block-sorting sign
    ],
)
def test_pairing_terms_pin_the_transposition_sign(second, expected):
    """No shipped model has p != q, so this is the only guard on _sort_blocks' sign.
    Drop that sign and both cases come out Hermitian -- a silent false positive."""
    op = FermionicSum([cdag(0) @ cdag(1), *second], 2)
    assert op.is_hermitian() is expected
    assert _is_hermitian_via_jw(op) is expected


def test_normal_order_on_a_hamiltonian_returns_a_plain_sum():
    """type(self)(...) here would re-enter FermionicHamiltonian's is_hermitian assertion."""
    from qaravan.core.hamiltonians import FermiHubbard1D

    canonical = FermiHubbard1D(2, t=1.0, U=2.0).normal_order()
    assert type(canonical) is FermionicSum
