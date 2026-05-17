"""
Trajectory simulation for a 1D collisional model.

Hamiltonian:  H = H_S + H_SA
  H_S   = Omega * sum_i X_i  +  V * sum_i Z_i Z_{i+1}   (TFI; class exists in hamiltonians.py)
  H_SA  = (gamma/2) * sum_k (X_{sys_k} X_{aux_k} + Y_{sys_k} Y_{aux_k})

Qubit layout — interleaved for MPS NN compatibility:
  sys_k = qubit 2k,   aux_k = qubit 2k+1   (k = 0, ..., NS-1)

Coupling B = {(2k, 2k+1)}: one aux per sys qubit, nearest-neighbor.

Initial state:  all sys in |1>,  all aux in |0>  (i.e. "1010..." bitstring)
Aux reset to:   |0>  after each measurement.

Gate order per cycle:  U = e^{-i H_SA} e^{-i H_S}
  => in gate_list: H_S gates first, H_SA gates second
  (Circuit.construct_layers will sequence them correctly since they share qubits.)

Analytical benchmark (Omega=0, V=0):
  Each (sys_k, aux_k) pair is independent.  Starting from |10>:
    e^{-it H_XY} |10> = cos(gamma*t)|10> - i sin(gamma*t)|01>
  Measuring aux in Z-basis:
    outcome '1'  with  p = sin^2(gamma*t)  => sys collapses to |0>  (absorbed)
    outcome '0'  with  1-p                 => sys stays in |1>
  Mean activity per aux at cycle m:
    E[k_m] = p * (1-p)^{m-1},    p = sin^2(gamma * t)
  Absorbing state: |0...0> (H_XY|00> = 0, no further evolution).

Note on reuse:  TFI(NS, jz=V, h=Omega) already builds H_S in hamiltonians.py.
  For larger systems, one could use TFI.layer_from_group to get Trotter gates on
  the sys qubits and then shift indices or embed into the full n=2*NS qubit space.
  Here we build the gates directly via expm for clarity.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import copy
from scipy.linalg import expm
from qaravan.core import *
from qaravan.tensorQ import *


# ─────────────────────────────────────────────
# Gate builders
# ─────────────────────────────────────────────

def xy_gate_matrix(gamma, t):
    """
    e^{-it * (gamma/2) * (XX + YY)} on a (sys, aux) pair.
    Acts non-trivially only in the {|01>, |10>} subspace:
      [[cos(gamma*t), -i sin(gamma*t)],
       [-i sin(gamma*t), cos(gamma*t)]]
    leaving |00> and |11> unchanged.
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    H = (gamma / 2) * (np.kron(X, X) + np.kron(Y, Y))
    return expm(-1j * t * H)


def rx_gate_matrix(Omega, t):
    """e^{-it * Omega * X} — single-qubit transverse-field evolution."""
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    return expm(-1j * t * Omega * X)


def zz_gate_matrix(V, t):
    """e^{-it * V * Z x Z} — two-qubit ZZ coupling gate."""
    Z = np.diag([1., -1.])
    H = V * np.kron(Z, Z).astype(complex)
    return expm(-1j * t * H)


def make_gate_layer(NS, gamma, t, Omega=0.0, V=0.0):
    """
    Gate list for one Trotter step of H = H_S + H_SA.
    Order: H_S gates first, then H_SA.  The circuit builder sequences
    them correctly (they share qubits so they can't be parallelised).

    ZZ coupling (V != 0) acts on sys pairs (2k, 2k+2) — NNN in the
    interleaved layout.  MPSSim handles NNN gates via lift_nnn_gate.
    """
    gate_list = []

    # H_S: single-site X field on sys qubits
    if abs(Omega) > 1e-12:
        rx = rx_gate_matrix(Omega, t)
        for k in range(NS):
            gate_list.append(Gate('RX', [2 * k], rx))

    # H_S: ZZ coupling between adjacent sys qubits (NNN in full layout)
    if abs(V) > 1e-12:
        zz = zz_gate_matrix(V, t)
        for k in range(NS - 1):
            gate_list.append(Gate('ZZ', [2 * k, 2 * (k + 1)], zz))

    # H_SA: XY coupling on NN (sys_k, aux_k) pairs
    xy = xy_gate_matrix(gamma, t)
    for k in range(NS):
        gate_list.append(Gate('XY', [2 * k, 2 * k + 1], xy))

    return gate_list


# ─────────────────────────────────────────────
# Kraus reset to |0>
# ─────────────────────────────────────────────

def kraus_reset_zero_sv(sv, bath_sites, outcomes, n):
    """
    Apply K_{b_i} = |0><b_i| to each bath site after sampling outcome b_i.
      b='0': K_0 = |0><0| = [[1,0],[0,0]]  (no-op; bath already |0>)
      b='1': K_1 = |0><1| = [[0,1],[0,0]]  (flips bath |1> -> |0>)
    Then renormalize.
    """
    for site, bit in zip(bath_sites, outcomes):
        K = (np.array([[1, 0], [0, 0]], dtype=complex) if bit == '0'
             else np.array([[0, 1], [0, 0]], dtype=complex))
        sv = op_action(K, [site], sv.reshape([2] * n), local_dim=2)
    sv = sv.reshape(2**n)
    return sv / np.linalg.norm(sv)


def kraus_reset_zero_mps(mps, bath_sites, outcomes):
    """
    Apply K_{b_i} = |0><b_i| to each bath site tensor in-place.
    site shape: (left_bond, right_bond, local_dim).
    result[l,r,i] = sum_j K[i,j] * site[l,r,j]
    Invalidate cached environments, then renormalize.
    """
    for site_idx, bit in zip(bath_sites, outcomes):
        K = (np.array([[1, 0], [0, 0]], dtype=complex) if bit == '0'
             else np.array([[0, 1], [0, 0]], dtype=complex))
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

def sv_run(NS, gamma, t, num_shots, num_cycles, Omega=0.0, V=0.0):
    """
    Statevector trajectory loop.
    Returns activities[num_shots, num_cycles], each entry = fraction of aux
    qubits measured in |1> (= activity k_m from the PDF).
    """
    n = 2 * NS
    bath_sites = [2 * k + 1 for k in range(NS)]
    gate_list = make_gate_layer(NS, gamma, t, Omega, V)
    init_sv = string_to_sv('10' * NS, 2).astype(complex)

    activities = np.zeros((num_shots, num_cycles))
    for shot in range(num_shots):
        sv = init_sv.copy()
        for cycle in range(num_cycles):
            for gate in gate_list:
                sv = op_action(
                    gate.matrix.astype(complex), gate.indices,
                    sv.reshape([2] * n), local_dim=2
                ).reshape(2**n)
            outcome = measure_sv(sv, bath_sites)
            activities[shot, cycle] = outcome.count('1') / NS   # normalise by NA=NS
            sv = kraus_reset_zero_sv(sv, bath_sites, outcome, n)
    return activities


def mps_run(NS, gamma, t, num_shots, num_cycles, Omega=0.0, V=0.0, max_dim=None):
    """
    MPS trajectory loop.
    Returns activities[num_shots, num_cycles] normalised by NA=NS.
    """
    n = 2 * NS
    bath_sites = [2 * k + 1 for k in range(NS)]
    gate_list = make_gate_layer(NS, gamma, t, Omega, V)
    init_mps = string_to_mps('10' * NS, 2)

    activities = np.zeros((num_shots, num_cycles))
    for shot in range(num_shots):
        mps = copy.deepcopy(init_mps)
        for cycle in range(num_cycles):
            mps = apply_layer_to_mps(mps, gate_list, n, max_dim=max_dim)
            outcome = mps.fast_measure(bath_sites)
            activities[shot, cycle] = outcome.count('1') / NS
            kraus_reset_zero_mps(mps, bath_sites, outcome)
    return activities


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_pure_damping_analytical():
    """
    Omega=0, V=0, NS=2: purely XY-driven amplitude damping.

    Each (sys_k, aux_k) pair is independent and starts in |10>.
    After the gate layer:
      e^{-it H_XY}|10> = cos(gamma*t)|10> - i sin(gamma*t)|01>
    Measuring aux in Z-basis with outcome b, then applying K_b = |0><b|:
      b='1' (prob p):    sys -> |0>  (absorbed, activity=1 for this aux)
      b='0' (prob 1-p):  sys stays |1>, next cycle identical dynamics
    => E[activity at cycle m] = p * (1-p)^{m-1},   p = sin^2(gamma * t)

    What this catches:
      - wrong xy_gate_matrix (e.g. sign error)
      - Kraus reset to |1> instead of |0> (activity would never decay to 0)
      - wrong bath site indices
      - broken normalization in multi-cycle loop
    What it doesn't catch:
      - correlated errors between SV and MPS representations
    """
    NS = 2; gamma = 0.5; t = 1.0; num_shots = 1000; num_cycles = 12

    p = np.sin(gamma * t) ** 2
    expected = p * (1 - p) ** np.arange(num_cycles)
    # Conservative sigma: uses Bernoulli(p_m) variance (overestimates for NA=2)
    sigma = np.sqrt(expected * (1 - expected) / num_shots)

    np.random.seed(42)
    acts = sv_run(NS, gamma, t, num_shots, num_cycles)
    mean_act = acts.mean(axis=0)

    assert np.all(np.abs(mean_act - expected) < 4 * sigma + 1e-9), (
        f"Activity profile mismatch:\n"
        f"  observed: {mean_act.round(4)}\n"
        f"  expected: {expected.round(4)}\n"
        f"  4σ tol:   {(4 * sigma).round(4)}"
    )
    print("PASS  test_pure_damping_analytical")
    print(f"  observed: {mean_act.round(3)}")
    print(f"  expected: {expected.round(3)}")


def test_absorbing_state():
    """
    Omega=0, V=0, large gamma*t: verify convergence to |0...0> absorbing state.
    H_XY|00> = 0, so once all sys qubits reach |0> there is no further evolution.

    With gamma*t = 1, p = sin^2(1) ~ 0.708.
    E[activity at cycle 20] = p*(1-p)^19 ~ 5e-5 — negligible.

    What this catches:
      - any reset error that re-excites sys qubits (activity won't decay)
      - failure to reset aux, causing spurious dynamics
    """
    NS = 2; gamma = 1.0; t = 1.0; num_shots = 300; num_cycles = 20

    np.random.seed(0)
    acts = sv_run(NS, gamma, t, num_shots, num_cycles)

    late_activity = acts[:, -5:].mean()   # average over last 5 cycles and all shots
    assert late_activity < 0.02, \
        f"Late-time activity should be ~0, got {late_activity:.5f}"
    print(f"PASS  test_absorbing_state  (late-time mean activity = {late_activity:.5f})")


def test_sv_mps_agree():
    """
    SV and MPS trajectory runs should produce the same mean activity.
    Uses Omega > 0 to exercise the X-field gate (non-trivial gate layer,
    both single-qubit and two-qubit gates present).

    Tolerance: 4 * joint sigma from the SV run variance.

    What this catches:
      - errors in apply_layer_to_mps (e.g. wrong gate indices for MPS)
      - MPS-specific errors in kraus_reset_zero_mps (wrong einsum axis)
    What it doesn't catch:
      - systematic errors shared between SV and MPS
    """
    NS = 2; gamma = 0.5; Omega = 0.4; t = 1.0; num_shots = 500; num_cycles = 8

    np.random.seed(7)
    acts_sv = sv_run(NS, gamma, t, num_shots, num_cycles, Omega=Omega)
    np.random.seed(7)
    acts_mps = mps_run(NS, gamma, t, num_shots, num_cycles, Omega=Omega)

    mean_sv, mean_mps = acts_sv.mean(), acts_mps.mean()
    joint_sigma = np.sqrt(acts_sv.var(ddof=1) / (num_shots * num_cycles)) * np.sqrt(2)
    tol = 4 * joint_sigma

    assert abs(mean_sv - mean_mps) < tol, (
        f"SV={mean_sv:.4f} vs MPS={mean_mps:.4f}, "
        f"diff={abs(mean_sv - mean_mps):.4f} > tol={tol:.4f}"
    )
    print(f"PASS  test_sv_mps_agree  (SV={mean_sv:.3f}, MPS={mean_mps:.3f})")


if __name__ == "__main__":
    print("=" * 60)
    print("1D Collisional model — TFI system + amplitude damping")
    print("  sys_k = qubit 2k,  aux_k = qubit 2k+1")
    print("  initial state: all sys |1>,  all aux |0>")
    print("  reset: aux -> |0> after each measurement")
    print("=" * 60)
    test_pure_damping_analytical()
    test_absorbing_state()
    test_sv_mps_agree()
    print("=" * 60)
    print("All tests passed.")
