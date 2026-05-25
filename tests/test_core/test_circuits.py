"""Tests for Circuit.bind/num_params and circuit generation functions."""

import numpy as np
import pytest

from qaravan.core.circuits import (
    Circuit,
    ghz_circuit,
    nn_pairs,
    rx_layer,
    ry_layer,
    rz_layer,
    rxx_layer,
    ryy_layer,
    rzz_layer,
)
from qaravan.core.gates import CNOT, H, MatrixGate, RX, RY, RZ, RXX, RYY, RZZ

H_MATRIX = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


# ---------------------------------------------------------------------------
# nn_pairs
# ---------------------------------------------------------------------------


def test_nn_pairs_open():
    assert nn_pairs(4) == [[0, 1], [1, 2], [2, 3]]


def test_nn_pairs_periodic():
    pairs = nn_pairs(4, periodic=True)
    assert pairs[-1] == [3, 0]
    assert len(pairs) == 4


def test_nn_pairs_two_sites():
    assert nn_pairs(2) == [[0, 1]]


def test_nn_pairs_one_site():
    assert nn_pairs(1) == []


def test_nn_pairs_periodic_small():
    assert nn_pairs(2, periodic=True) == [[0, 1], [1, 0]]


# ---------------------------------------------------------------------------
# ghz_circuit — structure
# ---------------------------------------------------------------------------


def test_ghz_circuit_len():
    assert len(ghz_circuit(4)) == 4


def test_ghz_circuit_first_gate_is_h():
    circ = ghz_circuit(3)
    assert isinstance(circ[0], H)
    assert circ[0].indices == [0]


def test_ghz_circuit_remaining_are_cnots():
    circ = ghz_circuit(4)
    for i in range(1, 4):
        assert isinstance(circ[i], CNOT)
        assert circ[i].indices == [i - 1, i]


def test_ghz_circuit_num_sites():
    assert ghz_circuit(5).num_sites == 5


def test_ghz_circuit_n1():
    circ = ghz_circuit(1)
    assert len(circ) == 1
    assert isinstance(circ[0], H)


# ---------------------------------------------------------------------------
# ghz_circuit — physics via to_matrix()
# ---------------------------------------------------------------------------


def test_ghz_n2_bell_state():
    psi_in = np.array([1, 0, 0, 0], dtype=complex)
    psi_out = ghz_circuit(2).to_matrix() @ psi_in
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(psi_out, expected)


def test_ghz_n3_state():
    psi_in = np.zeros(8, dtype=complex)
    psi_in[0] = 1.0
    psi_out = ghz_circuit(3).to_matrix() @ psi_in
    expected = np.zeros(8, dtype=complex)
    expected[0] = expected[7] = 1.0 / np.sqrt(2)
    assert np.allclose(psi_out, expected)


# ---------------------------------------------------------------------------
# Single-site rotation layers — structure
# ---------------------------------------------------------------------------


def test_rx_layer_num_gates():
    assert len(rx_layer(5)) == 5


def test_rx_layer_indices():
    circ = rx_layer(4)
    for i in range(4):
        assert circ[i].indices == [i]


def test_rx_layer_gate_types():
    circ = rx_layer(3)
    for i in range(3):
        assert isinstance(circ[i], RX)


def test_rx_layer_provided_params():
    params = [0.1, 0.5, 1.2]
    circ = rx_layer(3, params=params)
    for i in range(3):
        assert circ[i].params == pytest.approx((params[i],))


def test_ry_layer_gate_types():
    circ = ry_layer(3)
    for g in circ.gates:
        assert isinstance(g, RY)


def test_rz_layer_provided_params():
    params = [0.3, 0.7]
    circ = rz_layer(2, params=params)
    for i in range(2):
        assert circ[i].params == pytest.approx((params[i],))


def test_rz_layer_seed_reproducible():
    c1 = rz_layer(4, seed=42)
    c2 = rz_layer(4, seed=42)
    for i in range(4):
        assert c1[i].params == pytest.approx(c2[i].params)


def test_rz_layer_different_seeds():
    c1 = rz_layer(4, seed=0)
    c2 = rz_layer(4, seed=1)
    assert not all(c1[i].params == pytest.approx(c2[i].params) for i in range(4))


def test_rx_layer_zero_sites():
    assert len(rx_layer(0)) == 0


# ---------------------------------------------------------------------------
# Two-site rotation layers — structure
# ---------------------------------------------------------------------------


def test_rzz_layer_num_gates():
    skeleton = nn_pairs(4)
    assert len(rzz_layer(skeleton)) == 3


def test_rzz_layer_indices():
    skeleton = [[0, 1], [2, 3]]
    circ = rzz_layer(skeleton)
    assert circ[0].indices == [0, 1]
    assert circ[1].indices == [2, 3]


def test_rzz_layer_gate_types():
    circ = rzz_layer([[0, 1], [1, 2]])
    for g in circ.gates:
        assert isinstance(g, RZZ)


def test_rzz_layer_provided_params():
    params = [0.4, 0.9]
    circ = rzz_layer([[0, 1], [1, 2]], params=params)
    for i in range(2):
        assert circ[i].params == pytest.approx((params[i],))


def test_rxx_layer_gate_types():
    for g in rxx_layer([[0, 1]]).gates:
        assert isinstance(g, RXX)


def test_ryy_layer_gate_types():
    for g in ryy_layer([[0, 1]]).gates:
        assert isinstance(g, RYY)


def test_rzz_layer_empty_skeleton():
    assert len(rzz_layer([])) == 0


# ---------------------------------------------------------------------------
# Single-site layers — physics
# ---------------------------------------------------------------------------


def test_rx_layer_at_zero_is_identity():
    circ = rx_layer(2, params=[0.0, 0.0])
    assert np.allclose(circ.to_matrix(), np.eye(4))


def test_rzz_layer_diagonal():
    theta = 0.7
    circ = rzz_layer([[0, 1]], params=[theta])
    diag = np.diag(circ.to_matrix())
    expected = np.array(
        [np.exp(-1j * theta), np.exp(1j * theta), np.exp(1j * theta), np.exp(-1j * theta)]
    )
    assert np.allclose(diag, expected)


# ---------------------------------------------------------------------------
# Circuit.num_params
# ---------------------------------------------------------------------------


def test_num_params_all_parametric():
    circ = rx_layer(3, params=[0.1, 0.2, 0.3])
    assert circ.num_params == 3


def test_num_params_mixed():
    circ = Circuit([RX(0, 0.5), H(1), RZZ([1, 2], 0.3)])
    assert circ.num_params == 2


def test_num_params_no_parametric():
    circ = Circuit([H(0), CNOT([0, 1])])
    assert circ.num_params == 0


def test_num_params_empty_circuit():
    assert Circuit([]).num_params == 0


# ---------------------------------------------------------------------------
# Circuit.bind
# ---------------------------------------------------------------------------


def test_bind_updates_params():
    circ = rx_layer(3, params=[0.0, 0.0, 0.0])
    new_params = [0.1, 0.2, 0.3]
    bound = circ.bind(new_params)
    for i in range(3):
        assert bound[i].params == pytest.approx((new_params[i],))


def test_bind_wrong_count_raises():
    circ = rx_layer(3, params=[0.0, 0.0, 0.0])
    with pytest.raises(AssertionError):
        circ.bind([0.1, 0.2])


def test_bind_original_unchanged():
    params = [0.0, 0.0]
    circ = rx_layer(2, params=params)
    circ.bind([1.0, 2.0])
    assert circ[0].params == pytest.approx((0.0,))
    assert circ[1].params == pytest.approx((0.0,))


def test_bind_non_parametric_preserved():
    circ = Circuit([H(0), RX(1, 0.0), CNOT([0, 1])])
    bound = circ.bind([np.pi / 2])
    assert isinstance(bound[0], H)
    assert isinstance(bound[2], CNOT)


def test_bind_returns_circuit():
    circ = rx_layer(2, params=[0.0, 0.0])
    assert isinstance(circ.bind([1.0, 2.0]), Circuit)


def test_bind_physics():
    # rx_layer(1) initialised at 0, bind to π/2 → same matrix as RX(0, π/2)
    circ = rx_layer(1, params=[0.0])
    bound = circ.bind([np.pi / 2])
    from qaravan.core.gates import RX as _RX

    expected = _RX(0, np.pi / 2).matrix
    assert np.allclose(bound.to_matrix(), expected)
