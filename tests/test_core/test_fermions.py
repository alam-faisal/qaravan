"""Tests for core/fermions.py: FermionicSum algebra, Jordan-Wigner, Majorana operators."""

import numpy as np
import pytest
from qaravan.backends.statevector import Statevector
from qaravan.core.fermions import (
    FermionicEncoding,
    FermionicSum,
    c,
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
