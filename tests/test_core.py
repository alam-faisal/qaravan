import numpy as np
from qaravan.tensorQ import *
from qaravan.core import *

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

def test_contract_and_decimate():
    # Test contract_sites produces the correct shape for MPS sites
    sites = [np.random.rand(4,3,2)] + [np.random.rand(3,3,2)] * 1 + [np.random.rand(3,5,2)]
    c_site = contract_sites(sites)
    assert c_site.shape == (4, 5, 2**3), "contract_sites failed for MPS sites"

    # Test contract_sites produces the correct shape for MPDO sites
    sites = [np.random.rand(4,3,2,2)] + [np.random.rand(3,3,2,2)] * 1 + [np.random.rand(3,5,2,2)]
    c_site = contract_sites(sites)
    assert c_site.shape == (4, 5, 2**6), "contract_sites failed for MPDO sites"

    # Test decimate produces the correct shape and correct tensors for MPS sites
    sites = [np.random.rand(4,3,2)] + [np.random.rand(3,3,2)] * 1 + [np.random.rand(3,5,2)]
    c_site = contract_sites(sites)
    dec_sites = decimate(c_site, 2)
    c_site2 = contract_sites(dec_sites)
    assert np.allclose(c_site, c_site2), "decimate failed for MPS sites"
    assert all(s.shape == d.shape for s, d in zip(sites, dec_sites)), "Shapes do not match for MPS sites"

    # Test decimate produces the correct shape and correct tensors for MPDO sites
    sites = [np.random.rand(4,3,2,2)] + [np.random.rand(3,3,2,2)] * 1 + [np.random.rand(3,5,2,2)]
    c_site = contract_sites(sites)
    dec_sites = decimate(c_site, 2*2)  # super local_dim is square of local_dim
    dec_sites = [s.reshape(s.shape[0], s.shape[1], 2, 2) for s in dec_sites]
    c_site2 = contract_sites(dec_sites)
    assert np.allclose(c_site, c_site2), "decimate failed for MPDO sites"
    assert all(s.shape == d.shape for s, d in zip(sites, dec_sites)), "Shapes do not match for MPDO sites"