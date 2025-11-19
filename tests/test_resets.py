import numpy as np
from qaravan.core import *
from qaravan.tensorQ import *


def run_with_statevector(gate_list, n=None):
    """Helper to run circuit with StatevectorSim and return final state"""
    circ = Circuit(gate_list, n=n)
    sim = StatevectorSim(circ)
    sim.run(progress_bar=False)
    return sim.get_statevector()


def run_with_mps(gate_list, n=None):
    """Helper to run circuit with MPSSim and return final state"""
    circ = Circuit(gate_list, n=n)
    sim = MPSSim(circ)
    mps = sim.run(progress_bar=False)
    return mps.to_vector()


def test_reset_to_zero():
    """Test reset collapses superposition to |0⟩"""
    gate_list = [H(0), Reset(0, "0")]
    expected = np.array([1, 0])
    
    sv = run_with_statevector(gate_list)
    assert np.allclose(sv, expected), f"StatevectorSim: Expected |0⟩, got {sv}"
    
    sv = run_with_mps(gate_list)
    assert np.allclose(sv, expected), f"MPSSim: Expected |0⟩, got {sv}"
    
    print("✓ Passed: Reset to |0⟩")


def test_reset_to_one():
    """Test reset collapses superposition to |1⟩"""
    gate_list = [H(0), Reset(0, "1")]
    expected = np.array([0, 1])
    
    sv = run_with_statevector(gate_list)
    assert np.allclose(sv, expected), f"StatevectorSim: Expected |1⟩, got {sv}"
    
    sv = run_with_mps(gate_list)
    assert np.allclose(sv, expected), f"MPSSim: Expected |1⟩, got {sv}"
    
    print("✓ Passed: Reset to |1⟩")


def test_reset_preserves_other_qubits():
    """Test reset preserves separable qubits"""
    gate_list = [H(0), X(1), Reset(0, "1")]
    expected = np.array([0, 0, 0, 1])  # |11⟩
    
    sv = run_with_statevector(gate_list, n=2)
    assert np.allclose(sv, expected), f"StatevectorSim: Expected |11⟩, got {sv}"
    
    sv = run_with_mps(gate_list, n=2)
    assert np.allclose(sv, expected), f"MPSSim: Expected |11⟩, got {sv}"
    
    print("✓ Passed: Reset preserves other qubits")


def test_reset_breaks_entanglement():
    """Test reset breaks entanglement correctly"""
    gate_list = [H(0), CNOT([1,0]), Reset(0, "0")]
    expected = np.array([1, 0, 0, 0])
    
    sv = run_with_statevector(gate_list, n=2)
    assert np.allclose(sv, expected), f"StatevectorSim: Expected |00⟩, got {sv}"
    
    sv = run_with_mps(gate_list, n=2)
    assert np.allclose(sv, expected), f"MPSSim: Expected |00⟩, got {sv}"
    
    print("✓ Passed: Reset breaks entanglement")


def test_multi_qubit_reset():
    """Test multi-qubit reset"""
    gate_list = [H(0), H(1), Reset([0,1], "10")]
    expected = np.array([0, 0, 1, 0])  # |10⟩
    
    sv = run_with_statevector(gate_list, n=2)
    assert np.allclose(sv, expected), f"StatevectorSim: Expected |10⟩, got {sv}"
    
    sv = run_with_mps(gate_list, n=2)
    assert np.allclose(sv, expected), f"MPSSim: Expected |10⟩, got {sv}"
    
    print("✓ Passed: Multi-qubit reset")


def test_sequential_resets():
    """Test sequential resets"""
    gate_list = [H(0), Reset(0, "1"), H(0), Reset(0, "1")]
    expected = -np.array([0, 1])
    
    sv = run_with_statevector(gate_list)
    assert np.allclose(sv, expected), f"StatevectorSim: Expected -|1⟩, got {sv}"
    
    sv = run_with_mps(gate_list)
    assert np.allclose(sv, expected), f"MPSSim: Expected -|1⟩, got {sv}"
    
    print("✓ Passed: Sequential resets")


def test_reset_ancilla_reuse():
    """Test reset allows ancilla qubit reuse"""
    gate_list = [
        H(0), CNOT([1,0]),     # Create Bell pair
        Reset(1, "1"),          # Reset qubit 1 (ancilla)
        H(1),                   # Use ancilla again
        CNOT([2,1])             # Entangle with new qubit
    ]
    expected = (string_to_sv("100") - string_to_sv("111")) / np.sqrt(2)
    
    sv = run_with_statevector(gate_list, n=3)
    assert np.allclose(sv, expected), f"StatevectorSim: Expected superposition state, got {sv}"
    
    sv = run_with_mps(gate_list, n=3)
    assert np.allclose(sv, expected), f"MPSSim: Expected superposition state, got {sv}"
    
    print("✓ Passed: Reset allows ancilla reuse")
