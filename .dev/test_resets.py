"""
Barebones test of Method 1 (statevector trajectory) and Method 3 (MPS trajectory)
for reset dynamics, bypassing the broken Reset gate entirely.

Correct per-trajectory reset: after sampling outcome b_i for bath site i,
apply K_{b_i} = |1><b_i| which collapses AND resets in one operation:
  K_0 = |1><0| = [[0,0],[1,0]]   (b='0': maps |0>->|1>, kills |1> component)
  K_1 = |1><1| = [[0,0],[0,1]]   (b='1': maps |1>->|1>, kills |0> component)
Then renormalize.

Test model: n=2, qubit 0 = system, qubit 1 = bath.
Gate layer = [H(0), CNOT(target=1, control=0)].
Init state: |01> (system |0>, bath |1>).

Analytical result per cycle:
  |01> --H(0)--> (|01>+|11>)/sqrt(2) --CNOT([1,0])--> (|01>+|10>)/sqrt(2)
  Bath marginal = I/2 => activity = 0.5 every cycle, independent of prior outcomes.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import copy

from qaravan.core import *
from qaravan.tensorQ import *

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def kraus_reset_sv(sv, bath_sites, outcomes, n):
    """Apply K_{b_i} = |1><b_i| to each bath site, then renormalize. Returns 1D sv."""
    for site, bit in zip(bath_sites, outcomes):
        K = np.array([[0,0],[1,0]], dtype=complex) if bit == '0' \
            else np.array([[0,0],[0,1]], dtype=complex)
        sv = op_action(K, [site], sv.reshape([2]*n), local_dim=2)
    sv = sv.reshape(2**n)
    return sv / np.linalg.norm(sv)


def kraus_reset_mps(mps, bath_sites, outcomes):
    """Apply K_{b_i} to each bath site tensor in-place, then renormalize."""
    for site_idx, bit in zip(bath_sites, outcomes):
        K = np.array([[0,0],[1,0]], dtype=complex) if bit == '0' \
            else np.array([[0,0],[0,1]], dtype=complex)
        # site shape: (left_bond, right_bond, local_dim)
        mps.sites[site_idx] = np.einsum('ij,lrj->lri', K, mps.sites[site_idx])
    mps.right_envs = None
    mps.left_envs = None
    mps.normalize()


def apply_layer_to_mps(mps, gate_list, n, max_dim=None):
    """Apply a gate layer to an MPS by running a single-pass MPSSim."""
    sim = MPSSim(Circuit(gate_list, n=n), init_state=copy.deepcopy(mps), max_dim=max_dim)
    sim.run(progress_bar=False)
    return sim.state


# ─────────────────────────────────────────────
# Trajectory runners
# ─────────────────────────────────────────────

def method1_run(gate_list, bath_sites, n, num_shots, num_cycles, init_sv):
    """
    Statevector trajectory loop.
    Returns activities array of shape (num_shots, num_cycles).
    """
    activities = np.zeros((num_shots, num_cycles))
    for shot in range(num_shots):
        sv = init_sv.copy().astype(complex)
        for cycle in range(num_cycles):
            for gate in gate_list:
                sv = op_action(
                    gate.matrix.astype(complex), gate.indices,
                    sv.reshape([2]*n), local_dim=2
                ).reshape(2**n)
            outcome = measure_sv(sv, bath_sites)
            activities[shot, cycle] = outcome.count('0')
            sv = kraus_reset_sv(sv, bath_sites, outcome, n)
    return activities


def method3_run(gate_list, bath_sites, n, num_shots, num_cycles, init_mps, max_dim=None):
    """
    MPS trajectory loop.
    Returns activities array of shape (num_shots, num_cycles).
    """
    activities = np.zeros((num_shots, num_cycles))
    for shot in range(num_shots):
        mps = copy.deepcopy(init_mps)
        for cycle in range(num_cycles):
            mps = apply_layer_to_mps(mps, gate_list, n, max_dim=max_dim)
            outcome = mps.fast_measure(bath_sites)
            activities[shot, cycle] = outcome.count('0')
            kraus_reset_mps(mps, bath_sites, outcome)
    return activities


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_sv_bath_reset_to_one():
    """
    After one cycle, the bath qubit's 1-RDM should be |1><1| exactly,
    regardless of the sampled outcome.
    Catches: wrong K operator, normalization failure, wrong site index.
    Does NOT catch: wrong gate layer, wrong init state.
    """
    n = 2
    bath_sites = [1]
    gate_list = [H(0), CNOT([1, 0])]
    init_sv = string_to_sv("01", 2).astype(complex)

    # run a full cycle manually
    sv = init_sv.copy()
    for gate in gate_list:
        sv = op_action(gate.matrix.astype(complex), gate.indices,
                       sv.reshape([2]*n), local_dim=2).reshape(2**n)
    outcome = measure_sv(sv, bath_sites)
    sv = kraus_reset_sv(sv, bath_sites, outcome, n)

    bath_rdm = rdm_from_sv(sv, bath_sites)
    expected = np.array([[0, 0], [0, 1]], dtype=complex)
    assert np.allclose(bath_rdm, expected, atol=1e-10), \
        f"[SV] Bath 1-RDM after reset should be |1><1|, got:\n{bath_rdm}"
    print("PASS  test_sv_bath_reset_to_one")


def test_mps_bath_reset_to_one():
    """
    Same as above but for MPS. Uses mps.one_rdm to read back the bath 1-RDM.
    Catches: wrong einsum axis, failure to invalidate cached envs.
    """
    n = 2
    bath_sites = [1]
    gate_list = [H(0), CNOT([1, 0])]
    init_mps = string_to_mps("01", 2)

    mps = copy.deepcopy(init_mps)
    mps = apply_layer_to_mps(mps, gate_list, n)
    outcome = mps.fast_measure(bath_sites)
    kraus_reset_mps(mps, bath_sites, outcome)

    bath_rdm = mps.one_rdm(1)
    expected = np.array([[0, 0], [0, 1]], dtype=complex)
    assert np.allclose(bath_rdm, expected, atol=1e-6), \
        f"[MPS] Bath 1-RDM after reset should be |1><1|, got:\n{bath_rdm}"
    print("PASS  test_mps_bath_reset_to_one")


def test_sv_activity_statistics():
    """
    500 shots x 10 cycles: mean activity per cycle should be 0.5.
    sigma = sqrt(0.25/500) ~ 0.022; using 4-sigma tolerance.
    Catches: systematic bias in sampling or reset logic.
    Does NOT catch: exact probability values (only mean).
    """
    n = 2
    bath_sites = [1]
    gate_list = [H(0), CNOT([1, 0])]
    init_sv = string_to_sv("01", 2)
    num_shots, num_cycles = 500, 10

    np.random.seed(42)
    acts = method1_run(gate_list, bath_sites, n, num_shots, num_cycles, init_sv)

    mean_per_cycle = acts.mean(axis=0)
    sigma = np.sqrt(0.25 / num_shots)
    tol = 4 * sigma
    assert np.all(np.abs(mean_per_cycle - 0.5) < tol), \
        f"[SV] Activity per cycle outside 4σ: {mean_per_cycle} (tol={tol:.3f})"
    print(f"PASS  test_sv_activity_statistics  (mean/cycle={mean_per_cycle.mean():.3f})")


def test_mps_activity_statistics():
    """
    Same as above for MPS trajectories.
    Catches: bias in MPS fast_measure or kraus_reset_mps.
    """
    n = 2
    bath_sites = [1]
    gate_list = [H(0), CNOT([1, 0])]
    init_mps = string_to_mps("01", 2)
    num_shots, num_cycles = 500, 10

    np.random.seed(42)
    acts = method3_run(gate_list, bath_sites, n, num_shots, num_cycles, init_mps)

    mean_per_cycle = acts.mean(axis=0)
    sigma = np.sqrt(0.25 / num_shots)
    tol = 4 * sigma
    assert np.all(np.abs(mean_per_cycle - 0.5) < tol), \
        f"[MPS] Activity per cycle outside 4σ: {mean_per_cycle} (tol={tol:.3f})"
    print(f"PASS  test_mps_activity_statistics (mean/cycle={mean_per_cycle.mean():.3f})")


def test_sv_mps_agree():
    """
    Methods 1 and 3 should give the same mean activity (within joint sampling noise).
    Uses 500 shots each; joint 4-sigma tolerance ~ 0.063.
    Catches: systematic divergence between statevector and MPS representations.
    Does NOT catch: trajectory-level differences (only means compared).
    """
    n = 2
    bath_sites = [1]
    gate_list = [H(0), CNOT([1, 0])]
    init_sv = string_to_sv("01", 2)
    init_mps = string_to_mps("01", 2)
    num_shots, num_cycles = 500, 10

    np.random.seed(7)
    acts_sv = method1_run(gate_list, bath_sites, n, num_shots, num_cycles, init_sv)
    np.random.seed(7)
    acts_mps = method3_run(gate_list, bath_sites, n, num_shots, num_cycles, init_mps)

    mean_sv = acts_sv.mean()
    mean_mps = acts_mps.mean()
    # joint sigma: two independent estimates of p=0.5 over num_shots*num_cycles samples
    joint_sigma = np.sqrt(0.25 / (num_shots * num_cycles)) * np.sqrt(2)
    tol = 4 * joint_sigma
    assert abs(mean_sv - mean_mps) < tol, \
        f"SV mean={mean_sv:.4f} vs MPS mean={mean_mps:.4f} differ by >{tol:.4f}"
    print(f"PASS  test_sv_mps_agree            (SV={mean_sv:.3f}, MPS={mean_mps:.3f})")


if __name__ == "__main__":
    print("=" * 55)
    print("Reset dynamics — Method 1 (SV) and Method 3 (MPS)")
    print("=" * 55)
    test_sv_bath_reset_to_one()
    test_mps_bath_reset_to_one()
    test_sv_activity_statistics()
    test_mps_activity_statistics()
    test_sv_mps_agree()
    print("=" * 55)
    print("All tests passed.")
