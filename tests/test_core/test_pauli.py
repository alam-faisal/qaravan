"""Tests for core/pauli.py."""

import numpy as np
import pytest
import stim
from qaravan.core.observable import (
    Magnetization,
)
from qaravan.core.pauli import (
    PauliString,
    PauliSum,
    clifford_conjugate,
    clifford_to_single_z,
    pauli_relations,
)

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# PauliString
# ---------------------------------------------------------------------------


def test_pauli_string_single_qubit_matrices():
    assert np.allclose(PauliString("I").to_matrix(), _I)
    assert np.allclose(PauliString("X").to_matrix(), _X)
    assert np.allclose(PauliString("Y").to_matrix(), _Y)
    assert np.allclose(PauliString("Z").to_matrix(), _Z)


def test_pauli_string_two_qubit():
    assert np.allclose(PauliString("XZ").to_matrix(), np.kron(_X, _Z))
    assert np.allclose(PauliString("IX").to_matrix(), np.kron(_I, _X))


def test_pauli_string_coeff():
    assert np.allclose(PauliString("XZ", coeff=2.0).to_matrix(), 2.0 * np.kron(_X, _Z))
    assert np.allclose(PauliString("Z", coeff=0.5).to_matrix(), 0.5 * _Z)


def test_pauli_string_lowercase_accepted():
    assert np.allclose(PauliString("xz").to_matrix(), np.kron(_X, _Z))


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
    assert np.allclose(ps.to_matrix(), 2.0 * _X)


def test_pauli_string_add_two_strings():
    total = PauliString("IZ") + PauliString("ZI")
    assert isinstance(total, PauliSum)
    assert len(total.terms) == 2
    assert np.allclose(total.to_matrix(), np.kron(_I, _Z) + np.kron(_Z, _I))


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
    assert np.allclose(obs.to_matrix(), expected)


def test_pauli_sum_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        PauliSum([PauliString("X"), PauliString("XZ")])


def test_pauli_sum_empty_raises():
    with pytest.raises(ValueError):
        PauliSum([])


def test_pauli_sum_scalar_mul():
    obs = 0.5 * PauliSum([PauliString("IZ"), PauliString("ZI")])
    expected = 0.5 * (np.kron(_I, _Z) + np.kron(_Z, _I))
    assert np.allclose(obs.to_matrix(), expected)


def test_pauli_sum_add():
    a = PauliSum([PauliString("IZ")])
    b = PauliSum([PauliString("ZI")])
    c = a + b
    assert len(c.terms) == 2
    assert np.allclose(c.to_matrix(), np.kron(_I, _Z) + np.kron(_Z, _I))


def test_pauli_sum_as_pauli_sum_returns_self():
    obs = PauliSum([PauliString("IZ")])
    assert obs.as_pauli_sum() is obs


def test_pauli_string_as_pauli_sum():
    ps = PauliString("X")
    psum = ps.as_pauli_sum()
    assert isinstance(psum, PauliSum)
    assert len(psum.terms) == 1
    assert np.allclose(psum.to_matrix(), _X)


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
    assert np.allclose(composed.to_matrix(), a.to_matrix() @ b.to_matrix())


def test_compose_associative_on_random_triples():
    rng = np.random.default_rng(0)
    letters = "IXYZ"
    for _ in range(20):
        a, b, c = (
            PauliString("".join(rng.choice(list(letters), size=3))) for _ in range(3)
        )
        left = (a @ b) @ c
        right = a @ (b @ c)
        assert np.allclose(left.to_matrix(), right.to_matrix())
        assert np.allclose(
            left.to_matrix(), a.to_matrix() @ b.to_matrix() @ c.to_matrix()
        )


def test_compose_requires_equal_length():
    with pytest.raises(ValueError):
        PauliString("X").compose(PauliString("XZ"))


def test_compose_pauli_sum_distributes():
    xx = PauliSum(["XI", "IX"])
    z = PauliString("ZI")
    composed = z @ xx
    assert np.allclose(composed.to_matrix(), z.to_matrix() @ xx.to_matrix())


def test_compose_raising_lowering_gives_number_operator():
    # (X - iY)/2 @ (X + iY)/2 = (I - Z)/2 -- the number-operator identity (doc eq. 4).
    lowering = PauliSum([PauliString("X", 0.5), PauliString("Y", 0.5j)])  # (X+iY)/2
    raising = PauliSum([PauliString("X", 0.5), PauliString("Y", -0.5j)])  # (X-iY)/2
    number = raising.compose(lowering).simplify()
    expected = PauliSum([PauliString("I", 0.5), PauliString("Z", -0.5)])
    assert np.allclose(number.to_matrix(), expected.to_matrix())


def test_simplify_merges_duplicate_strings():
    ps = PauliSum(
        [PauliString("XI", 1.0), PauliString("XI", 2.0), PauliString("IY", 3.0)]
    )
    simplified = ps.simplify()
    assert len(simplified.terms) == 2
    assert np.allclose(simplified.to_matrix(), ps.to_matrix())


def test_simplify_drops_near_zero_terms():
    ps = PauliSum(
        [PauliString("XI", 1.0), PauliString("XI", -1.0), PauliString("IY", 3.0)]
    )
    simplified = ps.simplify()
    assert len(simplified.terms) == 1
    assert simplified.terms[0].string == "IY"


# ---------------------------------------------------------------------------
# The Pauli index form
# ---------------------------------------------------------------------------


def _dense_from_bitmasks(ps: PauliString) -> np.ndarray:
    """Rebuild the dense matrix from (x_mask, z_mask, n_Y) alone -- the reference for bitmasks."""
    x_mask, z_mask, n_y = ps.bitmasks()
    dim = 2**ps.num_sites
    out = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        phase = ps.coeff * (1j**n_y) * (-1.0) ** bin(col & z_mask).count("1")
        out[col ^ x_mask, col] = phase
    return out


@pytest.mark.parametrize("num_sites", [1, 2, 3, 4])
def test_bitmasks_reproduce_to_matrix(num_sites):
    """Catches a wrong Y phase, swapped x/z masks, and a reversed bit convention -- to_matrix's
    reduce(np.kron, ...) is the definition of that convention."""
    for index in range(4**num_sites):
        digits = np.base_repr(index, base=4).zfill(num_sites)
        string = "".join("IXYZ"[int(d)] for d in digits)
        ps = PauliString(string, coeff=0.5 - 0.25j)
        np.testing.assert_allclose(_dense_from_bitmasks(ps), ps.to_matrix(), atol=1e-14)


def _popcount_sector(num_sites: int, num_particles: int) -> np.ndarray:
    indices = np.arange(2**num_sites, dtype=np.int64)
    return np.flatnonzero(np.bitwise_count(indices) == num_particles)


def _hopping_sum(num_sites: int) -> PauliSum:
    """XX + YY hopping on every bond plus staggered Z fields: conserves total popcount."""
    terms = []
    for site in range(num_sites - 1):
        for pauli in ("X", "Y"):
            string = ["I"] * num_sites
            string[site] = string[site + 1] = pauli
            terms.append(PauliString("".join(string), coeff=0.5))
    for site in range(num_sites):
        string = ["I"] * num_sites
        string[site] = "Z"
        terms.append(PauliString("".join(string), coeff=0.3 * (-1) ** site))
    return PauliSum(terms)


@pytest.mark.parametrize("num_particles", range(7))
def test_to_sector_matrix_matches_dense_then_slice(num_particles):
    """Catches the drop-vs-reject decision inverting, a searchsorted off-by-one, and
    row/column transposition. The reference path shares no code with the index form."""
    num_sites = 6
    pauli_sum = _hopping_sum(num_sites)
    sector = _popcount_sector(num_sites, num_particles)
    reference = pauli_sum.to_matrix()[np.ix_(sector, sector)]
    np.testing.assert_allclose(
        pauli_sum.to_sector_matrix(sector), reference, atol=1e-12
    )


def test_a_term_may_leave_the_sector_without_raising():
    """A lone XX is hopping *plus* pairing: it maps some in-sector states out, so the drop
    branch is exercised and must not raise. Catches a future 'reject terms that leave the
    sector' guard, which is the natural-looking change that breaks the XX/YY cancellation."""
    sector = _popcount_sector(6, 3)
    lone = PauliSum([PauliString("XXIIII")])

    escaping = [j for j in sector if np.bitwise_count(j ^ 0b110000) != 3]
    assert escaping, "test is vacuous unless XX genuinely leaves the sector"

    np.testing.assert_allclose(
        lone.to_sector_matrix(sector),
        lone.to_matrix()[np.ix_(sector, sector)],
        atol=1e-12,
    )


def test_xx_and_yy_cancel_outside_the_sector():
    """The reason dropping is exact for a conserving sum: XX and YY share an x_mask, so their
    phases cancel on the pairing (out-of-sector) images and add on the hopping ones."""
    pairing = PauliSum([PauliString("XX", 1.0), PauliString("YY", 1.0)]).to_matrix()
    np.testing.assert_allclose(pairing[0b11, 0b00], 0.0, atol=1e-14)
    np.testing.assert_allclose(pairing[0b10, 0b01], 2.0, atol=1e-14)


def test_to_sector_matrix_is_hermitian_and_complex():
    """Catches a real-dtype accumulation silently discarding the i^n_Y phases."""
    sector = _popcount_sector(6, 3)
    matrix = _hopping_sum(6).to_sector_matrix(sector)
    assert matrix.dtype == np.complex128
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-14)


def test_magnetization_inherits_to_sector_matrix():
    """Pins the method on PauliSum rather than a specific constructor."""
    sector = _popcount_sector(4, 2)
    magnetization = Magnetization(4, "Z")
    np.testing.assert_allclose(
        magnetization.to_sector_matrix(sector),
        magnetization.to_matrix()[np.ix_(sector, sector)],
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Commuting Pauli groups
# ---------------------------------------------------------------------------


def _random_generators(num_qubits: int, num_generators: int) -> list[PauliString]:
    """Independent commuting +-1-signed strings: Z outputs of a random Clifford."""
    tableau = stim.Tableau.random(num_qubits)
    return [_from_stim(tableau.z_output(i)) for i in range(num_generators)]


def _from_stim(pauli: stim.PauliString) -> PauliString:
    return PauliString("".join("IXYZ"[k] for k in pauli), pauli.sign)


def _product(paulis: list[PauliString], num_qubits: int) -> PauliString:
    out = PauliString("I" * num_qubits)
    for pauli in paulis:
        out = out @ pauli
    return out


@pytest.mark.parametrize("num_generators", [1, 3, 5])
def test_clifford_to_single_z_sends_each_pauli_to_its_own_z(num_generators):
    """Catches a reversed tableau (stim's from_stabilizers maps Z_i to p_i, not the other way) and a lost sign:
    the image of p_i, sign included, is +Z_i."""
    paulis = _random_generators(5, num_generators)
    to_single_z = clifford_to_single_z(paulis)
    for i, pauli in enumerate(paulis):
        image = clifford_conjugate(PauliSum([pauli]), to_single_z).terms[0]
        expected = "I" * i + "Z" + "I" * (5 - i - 1)
        assert (image.string, image.coeff) == (expected, 1)


def test_clifford_conjugate_matches_dense_conjugation():
    """Catches sign or qubit-order mismatches between stim and Qaravan's big-endian matrices."""
    tableau = stim.Tableau.random(4)
    unitary = tableau.to_unitary_matrix(endian="big")
    pauli_sum = PauliSum(
        [PauliString("XYZI", 0.7), PauliString("ZZIY", -1.3j), PauliString("IIXX", 0.2)]
    )
    conjugated = clifford_conjugate(pauli_sum, tableau)
    np.testing.assert_allclose(
        conjugated.to_matrix(),
        unitary @ pauli_sum.to_matrix() @ unitary.conj().T,
        atol=1e-12,
    )


def test_pauli_relations_recovers_planted_products():
    """Planted signed products of random generators: each relation must multiply out exactly."""
    rng = np.random.default_rng(1)
    generators = _random_generators(6, 4)
    planted = []
    for _ in range(5):
        subset = [i for i in range(4) if rng.random() < 0.5]
        sign = rng.choice([1, -1])
        product = _product([generators[i] for i in subset], 6)
        planted.append(PauliString(product.string, sign * product.coeff))
    paulis = [
        generators[0],
        planted[0],
        generators[1],
        generators[2],
        *planted[1:],
        generators[3],
    ]
    independent, relations = pauli_relations(paulis)

    assert len(independent) == 4
    assert set(independent) | set(relations) == set(range(len(paulis)))
    for j, (subset, sign) in relations.items():
        assert all(i in independent and i < j for i in subset)
        product = _product([paulis[i] for i in subset], 6)
        assert (product.string, sign * product.coeff) == (
            paulis[j].string,
            paulis[j].coeff,
        )


def test_pauli_relations_reports_minus_identity():
    independent, relations = pauli_relations([PauliString("ZZ"), PauliString("ZZ", -1)])
    assert independent == [0]
    assert relations == {1: ([0], -1)}


def test_pauli_relations_rejects_anticommuting_paulis():
    with pytest.raises(ValueError, match="commute"):
        pauli_relations([PauliString("XI"), PauliString("ZI")])


def test_pauli_relations_rejects_non_unit_coefficients():
    with pytest.raises(ValueError, match="signed"):
        pauli_relations([PauliString("XI", 2.0)])
