"""Tests for environment_state_prep in applications/compilation.py."""

import numpy as np
import pytest

from qaravan.applications.circuit_library import brickwall_skeleton, two_local_circuit
from qaravan.applications.compilation import environment_state_prep
from qaravan.applications.run_context import RunContext
from qaravan.backends.statevector import Statevector
from qaravan.core.gates import is_unitary


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_environment_state_prep_circuit_and_skeleton_mutex():
    """Passing both circuit and skeleton raises ValueError."""
    n = 4
    skel = brickwall_skeleton(n) * 2
    circ = two_local_circuit(skel, seed=0)
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=1)
    with pytest.raises(ValueError):
        environment_state_prep(target, init, circuit=circ, skeleton=skel)


def test_environment_state_prep_neither_raises():
    """Passing neither circuit nor skeleton raises ValueError."""
    n = 4
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=1)
    with pytest.raises(ValueError):
        environment_state_prep(target, init)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_environment_state_prep_does_not_mutate_input_circuit():
    """The input circuit's gate matrices must be unchanged after the call."""
    n = 4
    skel = brickwall_skeleton(n) * 2
    circ = two_local_circuit(skel, seed=0)
    original_matrices = [g.matrix.copy() for g in circ.gates]
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=99)
    environment_state_prep(target, init, circuit=circ, context=RunContext(max_iter=3))
    for orig, gate in zip(original_matrices, circ.gates):
        np.testing.assert_array_equal(orig, gate.matrix)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------


def test_environment_state_prep_returns_unitary_gates():
    """Every gate in the returned circuit must be unitary."""
    n = 4
    skel = brickwall_skeleton(n) * 2
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=3)
    opt_circ, _ = environment_state_prep(
        target, init, skeleton=skel, context=RunContext(max_iter=5)
    )
    for gate in opt_circ.gates:
        assert is_unitary(gate.matrix, atol=1e-10), f"Gate {gate.name} is not unitary"


def test_environment_state_prep_cost_list_length():
    """cost_list length: 1 (initial) + sweeps × 2 × (n_gates - 1) gate updates."""
    n = 4
    skel = brickwall_skeleton(n) * 2  # 6 gates
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=7)
    n_sweeps = 3
    ctx = RunContext(max_iter=n_sweeps, stop_ratio=None, stop_absolute=None)
    _, cost_list = environment_state_prep(target, init, skeleton=skel, context=ctx)
    n_gates = len(skel)
    expected = 1 + n_sweeps * 2 * (n_gates - 1)
    assert len(cost_list) == expected


def test_environment_state_prep_initial_cost_in_list():
    """First entry of cost_list is the infidelity before any optimization."""
    n = 4
    skel = brickwall_skeleton(n) * 2
    circ = two_local_circuit(skel, seed=0)
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=7)
    ansatz = init.apply(circ)
    expected_initial = 1.0 - abs(target.overlap(ansatz))
    _, cost_list = environment_state_prep(
        target, init, circuit=circ, context=RunContext(max_iter=1)
    )
    assert np.isclose(cost_list[0], expected_initial, atol=1e-12)


def test_environment_state_prep_costs_non_negative():
    """Infidelity must be non-negative throughout."""
    n = 4
    skel = brickwall_skeleton(n) * 2
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=5)
    _, cost_list = environment_state_prep(
        target, init, skeleton=skel, context=RunContext(max_iter=10)
    )
    assert all(c >= -1e-10 for c in cost_list)


# ---------------------------------------------------------------------------
# Convergence (physics correctness)
# ---------------------------------------------------------------------------


def test_environment_state_prep_converges():
    """4-qubit random target: infidelity < 0.01 within 200 sweeps."""
    n = 4
    skel = brickwall_skeleton(n) * 2
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=42)
    ctx = RunContext(max_iter=200, stop_ratio=1e-10)
    _, cost_list = environment_state_prep(target, init, skeleton=skel, context=ctx)
    assert cost_list[-1] < 0.01, f"Final infidelity {cost_list[-1]:.4f} >= 0.01"


def test_environment_state_prep_final_cost_lower_than_initial():
    """Optimization must reduce infidelity."""
    n = 4
    skel = brickwall_skeleton(n) * 2
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=13)
    ctx = RunContext(max_iter=20)
    _, cost_list = environment_state_prep(target, init, skeleton=skel, context=ctx)
    assert cost_list[-1] < cost_list[0]


# ---------------------------------------------------------------------------
# RunContext stopping in context of environment_state_prep
# ---------------------------------------------------------------------------


def test_environment_state_prep_stops_at_max_iter():
    """Optimization runs exactly max_iter sweeps when no plateau."""
    n = 4
    skel = brickwall_skeleton(n) * 2
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=7)
    n_sweeps = 4
    ctx = RunContext(max_iter=n_sweeps, stop_ratio=None, stop_absolute=None)
    _, cost_list = environment_state_prep(target, init, skeleton=skel, context=ctx)
    n_gates = len(skel)
    assert len(cost_list) == 1 + n_sweeps * 2 * (n_gates - 1)


def test_environment_state_prep_stops_at_plateau():
    """Using a very lenient stop_ratio: optimization terminates early."""
    n = 4
    skel = brickwall_skeleton(n) * 2
    init = Statevector(bitstring="0" * n)
    target = Statevector(n, random_seed=42)
    # stop as soon as relative change drops below 50% (should trigger quickly)
    ctx = RunContext(max_iter=1000, stop_ratio=0.5, stop_absolute=None)
    _, cost_list = environment_state_prep(target, init, skeleton=skel, context=ctx)
    n_gates = len(skel)
    max_possible = 1 + 1000 * 2 * (n_gates - 1)
    assert len(cost_list) < max_possible
