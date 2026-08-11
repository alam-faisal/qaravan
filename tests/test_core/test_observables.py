"""Tests for core/observables.py."""

import numpy as np
import pytest
from qaravan.core.observables import PauliString, PauliSum, LocalOp, Magnetization

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# PauliString
# ---------------------------------------------------------------------------


def test_pauli_string_single_qubit_matrices():
    assert np.allclose(PauliString("I").matrix, _I)
    assert np.allclose(PauliString("X").matrix, _X)
    assert np.allclose(PauliString("Y").matrix, _Y)
    assert np.allclose(PauliString("Z").matrix, _Z)


def test_pauli_string_two_qubit():
    assert np.allclose(PauliString("XZ").matrix, np.kron(_X, _Z))
    assert np.allclose(PauliString("IX").matrix, np.kron(_I, _X))


def test_pauli_string_coeff():
    assert np.allclose(PauliString("XZ", coeff=2.0).matrix, 2.0 * np.kron(_X, _Z))
    assert np.allclose(PauliString("Z", coeff=0.5).matrix, 0.5 * _Z)


def test_pauli_string_lowercase_accepted():
    assert np.allclose(PauliString("xz").matrix, np.kron(_X, _Z))


def test_pauli_string_invalid_raises():
    with pytest.raises(ValueError):
        PauliString("XA")


def test_pauli_string_indices():
    ps = PauliString("XZI")
    assert ps.indices == [0, 1, 2]


# ---------------------------------------------------------------------------
# PauliString arithmetic
# ---------------------------------------------------------------------------


def test_pauli_string_scalar_mul():
    ps = 2.0 * PauliString("X")
    assert ps.coeff == 2.0
    assert np.allclose(ps.matrix, 2.0 * _X)


def test_pauli_string_add_two_strings():
    total = PauliString("IZ") + PauliString("ZI")
    assert isinstance(total, PauliSum)
    assert len(total.terms) == 2
    assert np.allclose(total.matrix, np.kron(_I, _Z) + np.kron(_Z, _I))


def test_pauli_string_add_string_and_sum():
    ps = PauliString("IZ")
    psum = PauliSum([PauliString("ZI"), PauliString("XX")])
    result = ps + psum
    assert isinstance(result, PauliSum)
    assert len(result.terms) == 3


# ---------------------------------------------------------------------------
# PauliSum
# ---------------------------------------------------------------------------


def test_pauli_sum_as_matrix():
    obs = PauliSum([PauliString("IZ"), PauliString("ZI")])
    expected = np.kron(_I, _Z) + np.kron(_Z, _I)
    assert np.allclose(obs.matrix, expected)


def test_pauli_sum_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        PauliSum([PauliString("X"), PauliString("XZ")])


def test_pauli_sum_empty_raises():
    with pytest.raises(ValueError):
        PauliSum([])


def test_pauli_sum_scalar_mul():
    obs = 0.5 * PauliSum([PauliString("IZ"), PauliString("ZI")])
    expected = 0.5 * (np.kron(_I, _Z) + np.kron(_Z, _I))
    assert np.allclose(obs.matrix, expected)


def test_pauli_sum_add():
    a = PauliSum([PauliString("IZ")])
    b = PauliSum([PauliString("ZI")])
    c = a + b
    assert len(c.terms) == 2
    assert np.allclose(c.matrix, np.kron(_I, _Z) + np.kron(_Z, _I))


def test_pauli_sum_as_pauli_sum_returns_self():
    obs = PauliSum([PauliString("IZ")])
    assert obs.as_pauli_sum() is obs


def test_pauli_string_as_pauli_sum():
    ps = PauliString("X")
    psum = ps.as_pauli_sum()
    assert isinstance(psum, PauliSum)
    assert len(psum.terms) == 1
    assert np.allclose(psum.matrix, _X)


# ---------------------------------------------------------------------------
# LocalOp
# ---------------------------------------------------------------------------


def test_local_op_matrix_returned_as_is():
    obs = LocalOp(_Z, [0])
    assert np.allclose(obs.matrix, _Z)


def test_local_op_indices_stored():
    obs = LocalOp(_Z, [1])
    assert obs.indices == [1]
    assert np.allclose(obs.matrix, _Z)  # still just Z, no embedding


def test_local_op_two_site():
    zz = np.kron(_Z, _Z)
    obs = LocalOp(zz, [0, 1])
    assert np.allclose(obs.matrix, zz)
    assert obs.indices == [0, 1]


# ---------------------------------------------------------------------------
# Magnetization
# ---------------------------------------------------------------------------


def test_magnetization_is_pauli_sum():
    assert isinstance(Magnetization(2), PauliSum)


def test_magnetization_matrix_shape():
    M = Magnetization(3)
    assert M.matrix.shape == (8, 8)


def test_magnetization_on_basis_states():
    M = Magnetization(2)
    mat = M.matrix
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
    assert np.isclose(plus @ M.matrix @ plus, 1.0)


def test_magnetization_invalid_axis():
    with pytest.raises(ValueError):
        Magnetization(2, axis="W")


# ---------------------------------------------------------------------------
# Operator composition (matrix product, not tensor product)
# ---------------------------------------------------------------------------


def _assert_pauli_string_equal(actual: PauliString, string: str, coeff: complex):
    assert actual.string == string
    assert np.isclose(actual.coeff, coeff)


def test_compose_single_qubit_identities():
    _assert_pauli_string_equal(PauliString("X").compose(PauliString("Y")), "Z", 1j)
    _assert_pauli_string_equal(PauliString("Y").compose(PauliString("X")), "Z", -1j)
    _assert_pauli_string_equal(PauliString("Y").compose(PauliString("Z")), "X", 1j)
    _assert_pauli_string_equal(PauliString("Z").compose(PauliString("X")), "Y", 1j)
    _assert_pauli_string_equal(PauliString("X").compose(PauliString("X")), "I", 1.0)


def test_compose_matmul_operator_matches_matrix_product():
    a, b = PauliString("XZ", coeff=0.5), PauliString("YZ", coeff=2.0)
    composed = a @ b
    assert np.allclose(composed.matrix, a.matrix @ b.matrix)


def test_compose_associative_on_random_triples():
    rng = np.random.default_rng(0)
    letters = "IXYZ"
    for _ in range(20):
        a, b, c = (PauliString("".join(rng.choice(list(letters), size=3))) for _ in range(3))
        left = (a @ b) @ c
        right = a @ (b @ c)
        assert np.allclose(left.matrix, right.matrix)
        assert np.allclose(left.matrix, a.matrix @ b.matrix @ c.matrix)


def test_compose_requires_equal_length():
    with pytest.raises(ValueError):
        PauliString("X").compose(PauliString("XZ"))


def test_compose_pauli_sum_distributes():
    xx = PauliSum(["XI", "IX"])
    z = PauliString("ZI")
    composed = z.compose(xx)
    assert np.allclose(composed.matrix, z.matrix @ xx.matrix)


def test_compose_raising_lowering_gives_number_operator():
    # (X - iY)/2 @ (X + iY)/2 = (I - Z)/2 -- the number-operator identity (doc eq. 4).
    lowering = PauliSum([PauliString("X", 0.5), PauliString("Y", 0.5j)])   # (X+iY)/2
    raising = PauliSum([PauliString("X", 0.5), PauliString("Y", -0.5j)])   # (X-iY)/2
    number = raising.compose(lowering).simplify()
    expected = PauliSum([PauliString("I", 0.5), PauliString("Z", -0.5)])
    assert np.allclose(number.matrix, expected.matrix)


def test_simplify_merges_duplicate_strings():
    ps = PauliSum([PauliString("XI", 1.0), PauliString("XI", 2.0), PauliString("IY", 3.0)])
    simplified = ps.simplify()
    assert len(simplified.terms) == 2
    assert np.allclose(simplified.matrix, ps.matrix)


def test_simplify_drops_near_zero_terms():
    ps = PauliSum([PauliString("XI", 1.0), PauliString("XI", -1.0), PauliString("IY", 3.0)])
    simplified = ps.simplify()
    assert len(simplified.terms) == 1
    assert simplified.terms[0].string == "IY"
