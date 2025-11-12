import numpy as np
from qaravan.tensorQ import *
from qaravan.core import *
from qaravan.applications import *
import torch
from collections import Counter

def test_statevector_sim():
    gate_list = [
        H(0),               
        CNOT([1, 0], 3),    
        CNOT([2, 1], 3)    
    ]
    circ = Circuit(gate_list, 3)
    
    sim = StatevectorSim(circ)
    sim.run()
    statevector = sim.get_statevector()

    expected = np.zeros(8, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[-1] = 1 / np.sqrt(2)

    assert np.allclose(statevector, expected, atol=1e-7), "Statevector is not the expected GHZ state"

def test_statevector_sim_pauli_expectation():
    gate_list = [
        H(0),               
        CNOT([1, 0], 3),    
        CNOT([2, 1], 3)    
    ]
    circ = Circuit(gate_list, 3)
    
    sim = StatevectorSim(circ)
    izz_exp = sim.pauli_expectation('izz')  # <Z_1 Z_2>
    assert abs(izz_exp - 1.0) < 1e-7, f"Expected <Z_1 Z_2> = 1, got {izz_exp}"

    izi_exp = sim.pauli_expectation('izi')  # <Z_1>
    assert abs(izi_exp - 0.0) < 1e-7, f"Expected <Z_1> = 0, got {izi_exp}"

    # TODO: add a non-trivial test case WITHOUT GHZ state 

def test_construct_unitary():
    in_strings = ['00', '01', '10']
    local_dim = 2

    unitary = random_unitary(local_dim**2)
    out_states = [unitary @ string_to_sv(in_str, local_dim) for in_str in in_strings]
    Q = construct_unitary(in_strings, out_states, local_dim)

    assert np.allclose(Q.conj().T @ Q, np.eye(4), atol=1e-7), "Q is not unitary (Qd Q != I)."
    assert np.allclose(Q @ Q.conj().T, np.eye(4), atol=1e-7), "Q is not unitary (Q Qd != I)."

    for st in in_strings:
        original = unitary @ string_to_sv(st, local_dim)
        constructed = Q @ string_to_sv(st, local_dim)
        assert np.allclose(constructed, original, atol=1e-7), (
            f"Constructed Q does not match reference unitary for input state '{st}'."
        )

def test_circuit_copy_with_autograd_gate():
    theta = torch.randn(1, dtype=torch.float64, requires_grad=True)
    mat = torch.matrix_exp(-1j * theta * torch.tensor([[0., 1.], [1., 0.]], dtype=torch.complex128))
    assert mat.requires_grad

    gate = Gate('Rx', [0], mat)
    circ = Circuit([gate], n=1)
    circ_copy = circ.copy()
    
    sv = torch.tensor([1.0, 0.0], dtype=torch.complex128)
    sv_out = torch.matmul(gate.matrix, sv)
    loss = torch.real(torch.dot(sv_out.conj(), sv_out))
    loss.backward()
    assert theta.grad is not None
    print("Test passed: Circuit with autograd-compatible gate copied successfully.")

def test_ti_rmps_generation():
    num_sites = 4
    chi = 2

    rmps = ti_rmps(num_sites, chi)
    ref = rmps.sites[0]
    for site in rmps.sites:
        assert np.allclose(site, ref), "Not all tensors in rmps.sites are the same"
    
    print("Test passed: TI-RMPS generation successful with translation invariant tensors.")

def test_measurement_on_statevector():
    sv = (
        np.sqrt(0.3) * string_to_sv('0001') +
        np.sqrt(0.45) * string_to_sv('1110') +
        np.sqrt(0.25) * string_to_sv('1011')
    )
    counts = Counter()
    num_samples = 5000
    for _ in range(num_samples):
        outcome = measure_sv(sv, [0, 1])
        counts[outcome] += 1

    assert abs(counts['00'] / num_samples - 0.3) < 0.05, f"Expected ~0.3, got {counts['00'] / num_samples}"
    assert abs(counts['11'] / num_samples - 0.45) < 0.05, f"Expected ~0.45, got {counts['11'] / num_samples}"
    assert abs(counts['10'] / num_samples - 0.25) < 0.05, f"Expected ~0.25, got {counts['10'] / num_samples}"

def test_measure_and_collapse_sv():
    sv = (
        np.sqrt(0.3) * string_to_sv('0001') +
        np.sqrt(0.45) * string_to_sv('1110') +
        np.sqrt(0.25) * string_to_sv('1011')
    )

    num_trials = 50
    for _ in range(num_trials):
        meas_indices = [0, 2]
        outcome, new_sv = measure_and_collapse_sv(sv, meas_indices)

        if outcome == '00':
            assert np.allclose(new_sv, string_to_sv('01')), f"Collapsed state mismatch for outcome '00'"
        elif outcome == '11':
            true_outcome = (
                np.sqrt(0.45 / (0.45 + 0.25)) * string_to_sv('10') +
                np.sqrt(0.25 / (0.45 + 0.25)) * string_to_sv('01')
            )
            assert np.allclose(new_sv, true_outcome, atol=1e-7), f"Collapsed state mismatch for outcome '11'"
        else:
            assert False, f"Unexpected measurement outcome: {outcome}"

def test_one_local_expectation(): 
    """ testing one_local_expectation() method of StatevectorSim """
    gate_list = [
        H(0),               
        CNOT([1, 0], 3),    
        CNOT([2, 1], 3)    
    ]
    circ = Circuit(gate_list, 3)
    sim = StatevectorSim(circ)
    z_exp = sim.one_local_expectation(pauli_Z, 2)  # <Z_2>
    assert abs(z_exp) < 1e-7, f"Expected <Z_2> = 0, got {z_exp}"