"""Tests for core/observable.py."""

import numpy as np
import pytest
from qaravan.backends.statevector import Statevector
from qaravan.core.observable import (
    HamiltonianTerm,
    LocalOp,
    Magnetization,
    NumberOperator,
    _embed_local_obs,
    embed_hamiltonian_terms,
    restrict_pauli_sum,
)
from qaravan.core.pauli import (
    PauliString,
    PauliSum,
)

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# LocalOp
# ---------------------------------------------------------------------------


def test_local_op_matrix_returned_as_is():
    obs = LocalOp(_Z, [0])
    assert np.allclose(obs.to_matrix(), _Z)


def test_local_op_indices_stored():
    obs = LocalOp(_Z, [1])
    assert obs.indices == [1]
    assert np.allclose(obs.to_matrix(), _Z)  # still just Z, no embedding


def test_local_op_two_site():
    zz = np.kron(_Z, _Z)
    obs = LocalOp(zz, [0, 1])
    assert np.allclose(obs.to_matrix(), zz)
    assert obs.indices == [0, 1]


# ---------------------------------------------------------------------------
# Magnetization
# ---------------------------------------------------------------------------


def test_magnetization_is_pauli_sum():
    assert isinstance(Magnetization(2), PauliSum)


def test_magnetization_matrix_shape():
    M = Magnetization(3)
    assert M.to_matrix().shape == (8, 8)


def test_magnetization_on_basis_states():
    M = Magnetization(2)
    mat = M.to_matrix()
    # |00⟩: both spins up → <M> = +1
    v00 = np.array([1, 0, 0, 0], dtype=complex)
    assert np.isclose(v00 @ mat @ v00, 1.0)
    # |11⟩: both spins down → <M> = -1
    v11 = np.array([0, 0, 0, 1], dtype=complex)
    assert np.isclose(v11 @ mat @ v11, -1.0)
    # |01⟩: one up, one down → <M> = 0
    v01 = np.array([0, 1, 0, 0], dtype=complex)
    assert np.isclose(v01 @ mat @ v01, 0.0)


def test_magnetization_axis_x():
    # |+⟩ eigenstate of X: <X> = 1
    M = Magnetization(1, axis="X")
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    assert np.isclose(plus @ M.to_matrix() @ plus, 1.0)


def test_magnetization_invalid_axis():
    with pytest.raises(ValueError):
        Magnetization(2, axis="W")


# ---------------------------------------------------------------------------
# _embed_local_obs / embed_hamiltonian_terms
# ---------------------------------------------------------------------------


def test_embed_local_obs_pauli_string_interior():
    ps = PauliString("ZZ", -1.0)
    [embedded] = _embed_local_obs(ps, [1, 2], 4)
    assert embedded.string == "IZZI"
    assert embedded.coeff == -1.0


def test_embed_local_obs_pauli_string_boundary():
    ps = PauliString("X", -0.5)
    [embedded] = _embed_local_obs(ps, [0], 3)
    assert embedded.string == "XII"
    assert embedded.coeff == -0.5


def test_embed_local_obs_pauli_string_last_site():
    ps = PauliString("Z", 2.0)
    [embedded] = _embed_local_obs(ps, [3], 4)
    assert embedded.string == "IIIZ"
    assert embedded.coeff == 2.0


def test_embed_local_obs_pauli_sum():
    obs = PauliSum([PauliString("XX", 1.0), PauliString("YY", 1.0)])
    result = _embed_local_obs(obs, [1, 2], 4)
    assert len(result) == 2
    assert result[0].string == "IXXI"
    assert result[1].string == "IYYI"


def test_embed_hamiltonian_terms_merges_and_simplifies():
    terms = [
        HamiltonianTerm([0, 1], PauliString("ZZ", 1.0)),
        HamiltonianTerm([1, 2], PauliString("ZZ", 1.0)),
    ]
    embedded = embed_hamiltonian_terms(terms, 3)
    assert isinstance(embedded, PauliSum)
    expected = PauliString("ZZI", 1.0).to_matrix() + PauliString("IZZ", 1.0).to_matrix()
    assert np.allclose(embedded.to_matrix(), expected)


def test_restrict_pauli_sum_is_inverse_of_embed():
    full = PauliSum([PauliString("IZZI", 0.5), PauliString("IZIY", -1.0)])
    term = restrict_pauli_sum(full)
    assert term.sites == [1, 2, 3]
    reembedded = embed_hamiltonian_terms([term], 4)
    assert np.allclose(reembedded.to_matrix(), full.simplify().to_matrix())


# ---------------------------------------------------------------------------
# NumberOperator
# ---------------------------------------------------------------------------


def test_number_operator_is_the_popcount_diagonal():
    """N is diagonal with popcount(i) as eigenvalues -- the hand-rolled form notebooks used."""
    for n in [1, 2, 3, 4]:
        diagonal = np.diag(NumberOperator(n).to_matrix()).real
        expected = [float(bin(i).count("1")) for i in range(2**n)]
        np.testing.assert_allclose(diagonal, expected, atol=1e-12)


def test_number_operator_is_diagonal():
    matrix = NumberOperator(3).to_matrix()
    np.testing.assert_allclose(matrix, np.diag(np.diag(matrix)), atol=1e-12)


def test_number_operator_on_a_qubit_subset_counts_only_those_qubits():
    # qubit q is character q of the bitstring, i.e. qubit 0 is the most significant bit
    n, qubits = 4, [0, 2]
    diagonal = np.diag(NumberOperator(n, qubits).to_matrix()).real
    for index in range(2**n):
        bits = format(index, f"0{n}b")
        assert diagonal[index] == pytest.approx(
            sum(bits[qubit] == "1" for qubit in qubits)
        ), bits


def test_number_operator_subsets_sum_to_the_total():
    n = 4
    up, down = NumberOperator(n, [0, 1]), NumberOperator(n, [2, 3])
    np.testing.assert_allclose(
        up.to_matrix() + down.to_matrix(), NumberOperator(n).to_matrix(), atol=1e-12
    )


def test_number_operator_expectation_of_a_basis_state():
    sv = Statevector(bitstring="1011")
    assert sv.expectation(NumberOperator(4)).real == pytest.approx(3.0, abs=1e-12)
    assert sv.expectation(NumberOperator(4, [0, 1])).real == pytest.approx(
        1.0, abs=1e-12
    )


def test_number_operator_rejects_bad_qubits():
    with pytest.raises(ValueError, match="must not repeat"):
        NumberOperator(3, [0, 0])
    with pytest.raises(ValueError, match=r"must lie in"):
        NumberOperator(3, [3])
