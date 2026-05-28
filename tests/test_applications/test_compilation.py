"""Tests for environment_state_prep and ghz_via_fusion in applications/compilation.py."""

import numpy as np
import pytest

from qaravan.applications.circuit_library import (
    brickwall_skeleton,
    ghz_cluster_prep_circuit,
    two_local_circuit,
)
from qaravan.applications.compilation import environment_state_prep, ghz_via_fusion
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


# ---------------------------------------------------------------------------
# ghz_via_fusion — input validation
# ---------------------------------------------------------------------------

INV_SQRT2 = 1.0 / np.sqrt(2)


def test_ghz_via_fusion_raises_for_k_le_2():
    """k <= 2 raises ValueError (degenerate case, formula undefined)."""
    with pytest.raises(ValueError):
        ghz_via_fusion(4, k=2)


def test_ghz_via_fusion_raises_for_non_integer_C():
    """(n-2) not divisible by (k-2) raises ValueError."""
    with pytest.raises(ValueError):
        ghz_via_fusion(5, k=4)  # C = (5-2)/(4-2) = 1.5


# ---------------------------------------------------------------------------
# ghz_via_fusion — output structure
# ---------------------------------------------------------------------------


def test_ghz_via_fusion_returns_correct_outcome_list_length():
    """len(outcomes) == C-1 == (n-2)//(k-2) - 1.

    Catches: loop-bounds bug (off-by-one in range(C-1)).
    """
    for n, k in [(4, 3), (6, 3)]:
        _, outcomes = ghz_via_fusion(n, k)
        C = (n - 2) // (k - 2)
        assert len(outcomes) == C - 1


# ---------------------------------------------------------------------------
# ghz_via_fusion — physics correctness (4-qubit)
# ---------------------------------------------------------------------------


def _ghz_fidelity(sv: Statevector, kept_sites: list[int]) -> float:
    """F = ⟨GHZ_n|ρ_kept|GHZ_n⟩ for the n=len(kept_sites) GHZ state."""
    n = len(kept_sites)
    dim = 2**n
    ghz = np.zeros(dim)
    ghz[0] = ghz[dim - 1] = INV_SQRT2
    rdm = sv.rdm(kept_sites)
    return float(np.real(ghz @ rdm @ ghz))


def test_ghz_via_fusion_4qubit_always_succeeds():
    """ghz_via_fusion(4, k=3): 20 runs, fidelity with |GHZ_4⟩ on kept sites = 1.0.

    4 qubits kept: {0,1} from cluster 0, {4,5} from cluster 1.
    Catches: wrong correction for any of the 4 outcomes.
    Does NOT catch: bugs that only manifest for specific random seeds.
    """
    kept = [0, 1, 4, 5]
    for _ in range(20):
        sv, _ = ghz_via_fusion(4, k=3)
        assert np.isclose(_ghz_fidelity(sv, kept), 1.0, atol=1e-10)


def test_ghz_via_fusion_4qubit_all_outcomes_covered():
    """100 runs of ghz_via_fusion(4, k=3) must yield all 4 outcome strings.

    Catches: correction logic that silently breaks for specific outcomes.
    """
    seen: set[str] = set()
    for _ in range(100):
        _, outcomes = ghz_via_fusion(4, k=3)
        seen.add(outcomes[0])
    assert seen == {"00", "01", "10", "11"}


# ---------------------------------------------------------------------------
# ghz_via_fusion — physics correctness (6-qubit, multi-boundary)
# ---------------------------------------------------------------------------


def test_ghz_via_fusion_6qubit_k3():
    """n=6, k=3: C=4 clusters, 3 fusions; 20 runs, fidelity 1.0 on kept sites.

    Kept sites: {0,1} (cluster 0), {4} (cluster 1), {7} (cluster 2), {10,11} (cluster 3).
    Catches: sequential decoder error for multi-boundary case.
    """
    kept = [0, 1, 4, 7, 10, 11]
    for _ in range(20):
        sv, outcomes = ghz_via_fusion(6, k=3)
        assert len(outcomes) == 3
        assert np.isclose(_ghz_fidelity(sv, kept), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# ghz_via_fusion — cluster independence (pre-fusion)
# ---------------------------------------------------------------------------


def test_ghz_cluster_prep_circuits_are_independent():
    """Before fusion: each cluster's rdm is the pure |GHZ_3⟩ state (no cross-entanglement).

    Catches: cross-cluster entanglement in the prep step (would give mixed rdm per cluster).
    """
    init = Statevector(bitstring="000000")
    sv = init.apply(
        ghz_cluster_prep_circuit([0, 1, 2], 6) + ghz_cluster_prep_circuit([3, 4, 5], 6)
    )

    ghz3 = np.zeros(8)
    ghz3[0] = ghz3[7] = INV_SQRT2
    expected_rdm = np.outer(ghz3, ghz3)

    np.testing.assert_allclose(sv.rdm([0, 1, 2]), expected_rdm, atol=1e-10)
    np.testing.assert_allclose(sv.rdm([3, 4, 5]), expected_rdm, atol=1e-10)
