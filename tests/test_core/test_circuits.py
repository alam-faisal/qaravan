"""Tests for Circuit.bind/num_params, construct_layers, and circuit generation functions."""

import numpy as np
import pytest

from qaravan.applications.circuit_library import (
    ghz_circuit,
    nn_pairs,
    rx_layer,
    ry_layer,
    rz_layer,
    rxx_layer,
    ryy_layer,
    rzz_layer,
    two_local_circuit,
)
from qaravan.backends.statevector import Statevector
from qaravan.core.circuits import Circuit
from qaravan.core.gates import CNOT, H, RX, RY, RXX, RYY, RZZ, X, Z

H_MATRIX = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
INV_SQRT2 = 1.0 / np.sqrt(2)


# ---------------------------------------------------------------------------
# construct_layers — structural tests
# ---------------------------------------------------------------------------


def test_construct_layers_empty():
    """Empty circuit → empty layer list."""
    circ = Circuit([], num_sites=4)
    circ.construct_layers()
    assert circ.layers == []


def test_construct_layers_single_gate():
    """Single gate → one layer containing that gate."""
    circ = Circuit([H(0)], num_sites=2)
    circ.construct_layers()
    assert len(circ.layers) == 1
    assert len(circ.layers[0]) == 1


def test_construct_layers_fully_disjoint_packs_into_one_layer():
    """Three gates on disjoint sites → all in one layer."""
    circ = Circuit([H(0), H(2), H(4)], num_sites=6)
    circ.construct_layers()
    assert len(circ.layers) == 1
    assert len(circ.layers[0]) == 3


def test_construct_layers_same_site_gates_are_sequential():
    """Three gates on the same site → three separate layers."""
    circ = Circuit([H(0), X(0), Z(0)], num_sites=1)
    circ.construct_layers()
    assert len(circ.layers) == 3


def test_construct_layers_chain_topology():
    """H(0), CNOT(0→1), CNOT(1→2) must occupy three layers, not two.

    Regression test for the bug where CNOT([1,2]) was placed in layer 0 alongside
    H([0]) because they don't share sites, ignoring the sequential dependency
    through CNOT([0,1]).
    """
    circ = Circuit([H(0), CNOT([0, 1]), CNOT([1, 2])], num_sites=3)
    circ.construct_layers()
    assert len(circ.layers) == 3
    assert circ.layers[0][0].name == "H"
    assert circ.layers[1][0].indices == [0, 1]
    assert circ.layers[2][0].indices == [1, 2]


def test_construct_layers_star_topology():
    """H(0), CNOT(0,1), CNOT(0,2): root 0 is touched every gate → 3 sequential layers."""
    circ = Circuit([H(0), CNOT([0, 1]), CNOT([0, 2])], num_sites=3)
    circ.construct_layers()
    assert len(circ.layers) == 3
    assert circ.layers[0][0].name == "H"
    assert circ.layers[1][0].indices == [0, 1]
    assert circ.layers[2][0].indices == [0, 2]


def test_construct_layers_brickwall_4site():
    """Standard brickwall (0,1),(2,3),(1,2),(3,4) packs into exactly 2 layers."""
    skel = [[0, 1], [2, 3], [1, 2], [3, 4]]
    circ = two_local_circuit(skel, seed=0)
    circ.construct_layers()
    assert len(circ.layers) == 2
    layer0_idx = sorted(tuple(g.indices) for g in circ.layers[0])
    layer1_idx = sorted(tuple(g.indices) for g in circ.layers[1])
    assert layer0_idx == [(0, 1), (2, 3)]
    assert layer1_idx == [(1, 2), (3, 4)]


def test_construct_layers_transitive_dependency():
    """A→{0}, B→{0,1}, C→{1,2}: C must land in layer 2, not layer 0.

    A and C share no sites, but C has a transitive dependency on A through B.
    """
    circ = Circuit([H(0), CNOT([0, 1]), CNOT([1, 2])], num_sites=3)
    circ.construct_layers()
    assert len(circ.layers) == 3
    # CNOT([1,2]) must be in the last layer
    assert circ.layers[-1][0].indices == [1, 2]


def test_construct_layers_two_independent_chains_interleave():
    """Two independent 3-deep chains can be packed into depth 3, not 6."""
    gates = [H(0), H(3), CNOT([0, 1]), CNOT([3, 4]), CNOT([1, 2]), CNOT([4, 5])]
    circ = Circuit(gates, num_sites=6)
    circ.construct_layers()
    assert len(circ.layers) == 3
    assert len(circ.layers[0]) == 2  # H(0), H(3)
    assert len(circ.layers[1]) == 2  # CNOT(0→1), CNOT(3→4)
    assert len(circ.layers[2]) == 2  # CNOT(1→2), CNOT(4→5)


def test_construct_layers_idempotent():
    """Calling construct_layers twice produces the same layer structure."""
    circ = Circuit([H(0), CNOT([0, 1]), CNOT([1, 2])], num_sites=3)
    circ.construct_layers()
    first = [[g.indices for g in layer] for layer in circ.layers]
    circ.construct_layers()
    second = [[g.indices for g in layer] for layer in circ.layers]
    assert first == second


def test_construct_layers_no_empty_layers():
    """No layer should ever be empty."""
    cases = [
        [H(0), CNOT([0, 1]), CNOT([1, 2])],
        [H(0), H(2), CNOT([0, 1])],
        [H(0), H(1), H(2), H(3)],
        [H(0), X(0), Z(0), H(1)],
    ]
    for gates_list in cases:
        circ = Circuit(gates_list, num_sites=4)
        circ.construct_layers()
        assert all(len(layer) > 0 for layer in circ.layers)


def test_construct_layers_preserves_gate_order_within_layer():
    """Gates assigned to the same layer keep their original left-to-right order."""
    circ = Circuit([H(0), H(2), H(4)], num_sites=6)
    circ.construct_layers()
    indices = [g.indices[0] for g in circ.layers[0]]
    assert indices == [0, 2, 4]


def test_construct_layers_all_gates_present_exactly_once():
    """Every gate appears exactly once across all layers (no duplicates, no drops)."""
    gates = [H(0), CNOT([0, 1]), H(2), CNOT([1, 2])]
    circ = Circuit(gates, num_sites=3)
    circ.construct_layers()
    flat = [g for layer in circ.layers for g in layer]
    assert len(flat) == 4
    for g in circ.gates:
        assert g in flat  # same object, not a copy


def test_construct_layers_non_contiguous_sites():
    """Gates on non-contiguous or reversed site indices get correct ordering."""
    # CNOT([2,0]) touches {0,2}; CNOT([0,1]) touches {0,1} — share site 0 → sequential
    circ = Circuit([CNOT([2, 0]), CNOT([0, 1])], num_sites=3)
    circ.construct_layers()
    assert len(circ.layers) == 2


# ---------------------------------------------------------------------------
# construct_layers — physics correctness
# ---------------------------------------------------------------------------


def test_construct_layers_chain_ghz_correct_state():
    """Chain GHZ via StatevectorSimulator produces (|000⟩+|111⟩)/√2 after fix.

    Before the fix, CNOT([1,2]) was reordered to layer 0, giving (|000⟩+|110⟩)/√2.
    This is the direct physics regression test.
    """
    circ = Circuit([H(0), CNOT([0, 1]), CNOT([1, 2])], num_sites=3)
    sv = Statevector(bitstring="000").apply(circ)
    arr = sv.to_array()
    assert np.isclose(arr[0], INV_SQRT2, atol=1e-10)  # |000⟩ amplitude
    assert np.isclose(arr[7], INV_SQRT2, atol=1e-10)  # |111⟩ amplitude
    assert np.isclose(
        arr[6], 0.0, atol=1e-10
    )  # |110⟩ must be zero (was non-zero before fix)


def test_construct_layers_deep_chain_correct_state():
    """5-qubit chain GHZ: H(0), CNOT(0,1), ..., CNOT(3,4) → (|00000⟩+|11111⟩)/√2."""
    n = 5
    gates = [H(0)] + [CNOT([i, i + 1]) for i in range(n - 1)]
    circ = Circuit(gates, num_sites=n)
    sv = Statevector(bitstring="0" * n).apply(circ)
    arr = sv.to_array()
    assert np.isclose(arr[0], INV_SQRT2, atol=1e-10)  # |00000⟩
    assert np.isclose(arr[2**n - 1], INV_SQRT2, atol=1e-10)  # |11111⟩
    assert np.isclose(np.sum(np.abs(arr) ** 2), 1.0, atol=1e-10)


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
        [
            np.exp(-1j * theta),
            np.exp(1j * theta),
            np.exp(1j * theta),
            np.exp(-1j * theta),
        ]
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
    expected = RX(0, np.pi / 2).matrix
    assert np.allclose(bound.to_matrix(), expected)
