"""Tests for applications/quantum_info.py."""

import numpy as np
import pytest
from qaravan.applications.circuit_library import ghz_circuit
from qaravan.applications.quantum_info import (
    entanglement_depth,
    entanglement_entropy,
    entanglement_profile,
    phase_free_distance,
)
from qaravan.backends.statevector import Statevector
from qaravan.core.circuit import Circuit
from qaravan.core.gates import MatrixGate

# ---------------------------------------------------------------------------
# entanglement_entropy
# ---------------------------------------------------------------------------


def test_entanglement_entropy_product_state():
    """Product state |000⟩ has S=0 for any single-site cut."""
    sv = Statevector(bitstring="000")
    assert entanglement_entropy(sv, [0]) == pytest.approx(0.0, abs=1e-12)


def test_entanglement_entropy_product_state_multi_site_cut():
    """Product state has S=0 even for a multi-site subsystem."""
    sv = Statevector(bitstring="0000")
    assert entanglement_entropy(sv, [0, 1]) == pytest.approx(0.0, abs=1e-12)


def test_entanglement_entropy_bell_state():
    """Bell state |00⟩+|11⟩ / √2 has S = 1 bit for the [0]|[1] cut."""
    sv = Statevector(bitstring="00").apply(ghz_circuit(2))
    assert entanglement_entropy(sv, [0]) == pytest.approx(1.0, abs=1e-10)


def test_entanglement_entropy_bell_state_symmetric():
    """Entropy of [1] subsystem equals entropy of [0] subsystem for Bell state."""
    sv = Statevector(bitstring="00").apply(ghz_circuit(2))
    s0 = entanglement_entropy(sv, [0])
    s1 = entanglement_entropy(sv, [1])
    assert s0 == pytest.approx(s1, abs=1e-10)


def test_entanglement_entropy_full_subsystem_is_zero():
    """Tracing over nothing (all sites kept) gives pure state entropy = 0."""
    sv = Statevector(bitstring="00").apply(ghz_circuit(2))
    assert entanglement_entropy(sv, [0, 1]) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# entanglement_profile
# ---------------------------------------------------------------------------


def test_entanglement_profile_length():
    """Profile has length n-1 for an n-qubit state."""
    for n in [2, 4, 6]:
        sv = Statevector(bitstring="0" * n)
        profile = entanglement_profile(sv)
        assert len(profile) == n - 1


def test_entanglement_profile_product_state_all_zero():
    """Product state has zero entropy at every cut."""
    sv = Statevector(bitstring="0000")
    profile = entanglement_profile(sv)
    np.testing.assert_allclose(profile, 0.0, atol=1e-12)


def test_entanglement_profile_ghz():
    """GHZ on n qubits: every bipartition [0..k] | [k+1..n-1] has S = 1 bit."""
    n = 4
    sv = Statevector(bitstring="0" * n).apply(ghz_circuit(n))
    profile = entanglement_profile(sv)
    np.testing.assert_allclose(profile, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# entanglement_depth
# ---------------------------------------------------------------------------


def test_entanglement_depth_product():
    """Product state has entanglement depth 0."""
    sv = Statevector(bitstring="0000")
    assert entanglement_depth(sv) == 0


def test_entanglement_depth_ghz():
    """GHZ state has entanglement depth >= 1."""
    sv = Statevector(bitstring="0000").apply(ghz_circuit(4))
    assert entanglement_depth(sv) >= 1


def test_entanglement_depth_formula():
    """Depth equals ceil(max_S) for a known state.

    Bell state on 2 qubits: max_S = 1 bit, so depth = ceil(1.0) = 1.
    """
    sv = Statevector(bitstring="00").apply(ghz_circuit(2))
    assert entanglement_depth(sv) == 1


# ---------------------------------------------------------------------------
# phase_free_distance
# ---------------------------------------------------------------------------


def test_phase_free_distance_ignores_global_phase():
    """A global phase on either argument leaves the distance at zero."""
    sv = Statevector(3, random_seed=1)
    rotated = Statevector(array=np.exp(0.7j) * sv.to_array())
    assert phase_free_distance(sv, rotated) == pytest.approx(0.0, abs=1e-12)


def test_phase_free_distance_matches_overlap_formula():
    """For unit vectors the distance is sqrt(2 - 2|<b|a>|)."""
    a, b = Statevector(3, random_seed=1), Statevector(3, random_seed=2)
    overlap = abs(np.vdot(b.to_array(), a.to_array()))
    assert phase_free_distance(a, b) == pytest.approx(np.sqrt(2 - 2 * overlap), abs=1e-12)


def test_phase_free_distance_orthogonal_states():
    """Orthogonal unit vectors sit at sqrt(2) whatever the phase."""
    assert phase_free_distance(Statevector(bitstring="01"), Statevector(bitstring="10")) == pytest.approx(np.sqrt(2))


def test_phase_free_distance_accepts_raw_arrays():
    a, b = Statevector(2, random_seed=3), Statevector(2, random_seed=4)
    assert phase_free_distance(a.to_array(), b.to_array()) == pytest.approx(phase_free_distance(a, b))


def test_phase_free_distance_rejects_mixed_types():
    a, b = Statevector(2, random_seed=3), Statevector(2, random_seed=4)
    with pytest.raises(TypeError, match="same type"):
        phase_free_distance(a, b.to_array())


def test_phase_free_distance_unitaries_and_circuits():
    """Matrices (and circuits, via to_matrix) differing by a phase are at distance zero; X vs Z is not."""
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    assert phase_free_distance(np.exp(0.3j) * x, x) == pytest.approx(0.0, abs=1e-12)
    assert phase_free_distance(x, z) > 1.0
    circuit = Circuit([MatrixGate("x", [0], x)], num_sites=1)
    other = Circuit([MatrixGate("x", [0], np.exp(1.1j) * x)], num_sites=1)
    assert phase_free_distance(circuit, other) == pytest.approx(0.0, abs=1e-12)


def test_phase_free_distance_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        phase_free_distance(Statevector(2, random_seed=0), Statevector(3, random_seed=0))
