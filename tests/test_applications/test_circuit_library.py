"""Tests for brickwall_skeleton and two_local_circuit in applications/circuit_library.py."""

import numpy as np

from qaravan.applications.circuit_library import (
    bell_basis_circuit,
    brickwall_skeleton,
    ghz_cluster_prep_circuit,
    two_local_circuit,
)
from qaravan.backends.statevector import Statevector
from qaravan.core.gates import is_unitary


# ---------------------------------------------------------------------------
# brickwall_skeleton — 1D
# ---------------------------------------------------------------------------


def test_brickwall_1d_even_sites():
    """n=6: even-x pairs (0,1),(2,3),(4,5) then odd-x (1,2),(3,4) = 5 edges."""
    skel = brickwall_skeleton(6)
    assert len(skel) == 5
    assert [0, 1] in skel
    assert [2, 3] in skel
    assert [4, 5] in skel
    assert [1, 2] in skel
    assert [3, 4] in skel


def test_brickwall_1d_ordering():
    """Even-x edges come before odd-x edges."""
    skel = brickwall_skeleton(6)
    even_x = [[0, 1], [2, 3], [4, 5]]
    odd_x = [[1, 2], [3, 4]]
    even_idx = max(skel.index(e) for e in even_x)
    odd_idx = min(skel.index(o) for o in odd_x)
    assert even_idx < odd_idx


def test_brickwall_1d_odd_sites():
    """n=5: even-x pairs (0,1),(2,3) then odd-x (1,2),(3,4) = 4 edges."""
    skel = brickwall_skeleton(5)
    assert len(skel) == 4
    assert [0, 1] in skel
    assert [2, 3] in skel
    assert [1, 2] in skel
    assert [3, 4] in skel


def test_brickwall_1d_single_site_is_empty():
    assert brickwall_skeleton(1) == []


def test_brickwall_1d_two_sites():
    assert brickwall_skeleton(2) == [[0, 1]]


# ---------------------------------------------------------------------------
# brickwall_skeleton — 2D
# ---------------------------------------------------------------------------


def test_brickwall_2d_shape():
    """3×2 grid: 4 h-edges + 3 v-edges = 7 total."""
    skel = brickwall_skeleton(3, 2)
    assert len(skel) == 7


def test_brickwall_2d_contains_expected_edges():
    """3×2 grid (sites 0..5): verify each edge type is present."""
    skel = brickwall_skeleton(3, 2)
    pairs = [tuple(e) for e in skel]
    # horizontal edges
    assert (0, 1) in pairs  # even-x row 0
    assert (3, 4) in pairs  # even-x row 1
    assert (1, 2) in pairs  # odd-x row 0
    assert (4, 5) in pairs  # odd-x row 1
    # vertical edges: site (x,y)→(x,y+1), Lx=3
    # (0,0)→(0,1): 0→3, even-y
    assert (0, 3) in pairs
    # (1,0)→(1,1): 1→4, even-y
    assert (1, 4) in pairs
    # (2,0)→(2,1): 2→5, even-y
    assert (2, 5) in pairs


def test_brickwall_2d_ordering():
    """h-edges precede v-edges."""
    skel = brickwall_skeleton(3, 2)
    # last h-edge index < first v-edge index
    h_edges = [(0, 1), (3, 4), (1, 2), (4, 5)]
    v_edges = [(0, 3), (1, 4), (2, 5)]
    pairs = [tuple(e) for e in skel]
    last_h = max(pairs.index(e) for e in h_edges)
    first_v = min(pairs.index(e) for e in v_edges)
    assert last_h < first_v


def test_brickwall_2d_ly1_matches_1d():
    """brickwall_skeleton(n, 1) == brickwall_skeleton(n)."""
    for n in [4, 5, 6]:
        assert brickwall_skeleton(n, 1) == brickwall_skeleton(n)


def test_brickwall_depth_via_repetition():
    """Depth-2 circuit: repeat skeleton twice."""
    skel = brickwall_skeleton(4)
    deep = skel * 2
    assert len(deep) == 2 * len(skel)
    assert deep[: len(skel)] == skel


# ---------------------------------------------------------------------------
# two_local_circuit
# ---------------------------------------------------------------------------


def test_two_local_circuit_gate_count():
    skel = brickwall_skeleton(4) * 2
    circ = two_local_circuit(skel, seed=0)
    assert len(circ.gates) == len(skel)


def test_two_local_circuit_gate_indices():
    skel = brickwall_skeleton(4) * 2
    circ = two_local_circuit(skel, seed=0)
    for gate, pair in zip(circ.gates, skel):
        assert gate.indices == list(pair)


def test_two_local_circuit_matrices_unitary():
    skel = brickwall_skeleton(4) * 2
    circ = two_local_circuit(skel, seed=0)
    for gate in circ.gates:
        assert gate.matrix.shape == (4, 4)
        assert is_unitary(gate.matrix, atol=1e-10)


def test_two_local_circuit_reproducible():
    skel = brickwall_skeleton(4)
    c1 = two_local_circuit(skel, seed=42)
    c2 = two_local_circuit(skel, seed=42)
    for g1, g2 in zip(c1.gates, c2.gates):
        np.testing.assert_array_equal(g1.matrix, g2.matrix)


def test_two_local_circuit_different_seeds_differ():
    skel = brickwall_skeleton(4)
    c1 = two_local_circuit(skel, seed=0)
    c2 = two_local_circuit(skel, seed=1)
    assert not np.allclose(c1.gates[0].matrix, c2.gates[0].matrix)


def test_two_local_circuit_num_sites():
    skel = brickwall_skeleton(6)
    circ = two_local_circuit(skel, seed=0)
    assert circ.num_sites == 6


def test_two_local_circuit_non_contiguous_sites():
    """Custom skeleton with non-adjacent indices."""
    skel = [[0, 3], [1, 4], [2, 5]]
    circ = two_local_circuit(skel, seed=0)
    assert circ.num_sites == 6
    for gate, pair in zip(circ.gates, skel):
        assert gate.indices == pair
        assert is_unitary(gate.matrix)


# ---------------------------------------------------------------------------
# bell_basis_circuit
# ---------------------------------------------------------------------------

INV_SQRT2 = 1.0 / np.sqrt(2)

# Bell states as flat statevector arrays (index = qubit0*2 + qubit1)
# |Φ+⟩ = (|00⟩+|11⟩)/√2, |Φ-⟩ = (|00⟩-|11⟩)/√2
# |Ψ+⟩ = (|01⟩+|10⟩)/√2, |Ψ-⟩ = (|01⟩-|10⟩)/√2
_BELL_ARRAYS = [
    np.array([INV_SQRT2, 0, 0, INV_SQRT2]),
    np.array([INV_SQRT2, 0, 0, -INV_SQRT2]),
    np.array([0, INV_SQRT2, INV_SQRT2, 0]),
    np.array([0, INV_SQRT2, -INV_SQRT2, 0]),
]
_COMP_BITSTRINGS = ["00", "10", "01", "11"]


def test_bell_basis_circuit_maps_bell_states_to_comp_basis():
    """CNOT(0→1), H(0) maps each Bell state to a definite computational basis state.

    Catches: wrong CNOT/H order, wrong site assignment.
    Does NOT catch: multi-qubit system qubit-labeling errors (2-qubit test only).
    """
    circ = bell_basis_circuit(0, 1, 2)
    for arr, bs in zip(_BELL_ARRAYS, _COMP_BITSTRINGS):
        result = Statevector(array=arr).apply(circ)
        expected = Statevector(bitstring=bs)
        assert np.isclose(abs(result.overlap(expected)), 1.0, atol=1e-10)


def test_bell_basis_circuit_num_sites():
    """bell_basis_circuit(2, 3, 6) embeds in a 6-qubit register."""
    circ = bell_basis_circuit(2, 3, 6)
    assert circ.num_sites == 6


# ---------------------------------------------------------------------------
# ghz_cluster_prep_circuit
# ---------------------------------------------------------------------------


def test_ghz_cluster_prep_circuit_contiguous_gives_ghz_state():
    """cluster_sites=[0,1,2], num_sites=3 → state (|000⟩+|111⟩)/√2.

    Catches: implementation bugs in ghz_cluster_prep_circuit.
    """
    init = Statevector(bitstring="000")
    sv = init.apply(ghz_cluster_prep_circuit([0, 1, 2], 3))
    ghz3 = np.zeros(8)
    ghz3[0] = ghz3[7] = INV_SQRT2
    assert np.isclose(abs(sv.overlap(Statevector(array=ghz3))), 1.0, atol=1e-10)


def test_ghz_cluster_prep_circuit_offset_sites():
    """cluster_sites=[3,4,5], num_sites=6: qubits 0-2 are |000⟩; qubits 3-5 are GHZ.

    Catches: site-offset errors (applying gate to wrong qubit index).
    """
    init = Statevector(bitstring="000000")
    sv = init.apply(ghz_cluster_prep_circuit([3, 4, 5], 6))

    # qubits 0-2 should be |000⟩
    rdm_012 = sv.rdm([0, 1, 2])
    expected_vac = np.zeros((8, 8))
    expected_vac[0, 0] = 1.0
    np.testing.assert_allclose(rdm_012, expected_vac, atol=1e-10)

    # qubits 3-5 should be |GHZ_3⟩ = (|000⟩+|111⟩)/√2
    rdm_345 = sv.rdm([3, 4, 5])
    ghz3 = np.zeros(8)
    ghz3[0] = ghz3[7] = INV_SQRT2
    np.testing.assert_allclose(rdm_345, np.outer(ghz3, ghz3), atol=1e-10)
