"""Tests for applications/quantum_info.py."""

import numpy as np
import pytest

from qaravan.applications.circuit_library import ghz_circuit
from qaravan.applications.quantum_info import (
    entanglement_depth,
    entanglement_entropy,
    entanglement_profile,
)
from qaravan.backends.statevector import Statevector


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
    """Bell state |00⟩+|11⟩ / √2 has S = log(2) for the [0]|[1] cut."""
    sv = Statevector(bitstring="00").apply(ghz_circuit(2))
    assert entanglement_entropy(sv, [0]) == pytest.approx(np.log(2), abs=1e-10)


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
    """GHZ on n qubits: every bipartition [0..k] | [k+1..n-1] has S = log(2)."""
    n = 4
    sv = Statevector(bitstring="0" * n).apply(ghz_circuit(n))
    profile = entanglement_profile(sv)
    np.testing.assert_allclose(profile, np.log(2), atol=1e-10)


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
    """Depth equals ceil(max_S / log2) for a known state.

    Bell state on 2 qubits: max_S = log(2), so depth = ceil(1.0) = 1.
    """
    sv = Statevector(bitstring="00").apply(ghz_circuit(2))
    assert entanglement_depth(sv) == 1
