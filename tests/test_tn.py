import numpy as np
from qaravan.tensorQ import *
from qaravan.core import *

def test_mps_to_sv_and_back():
    """Test MPS to statevector conversion and back"""
    n = 4
    sv = random_sv(n)
    mps = sv_to_mps(sv)
    assert mps.num_sites == n, f"Expected {n} sites, got {mps.num_sites}"
    
    sv2 = mps.to_vector()
    assert np.allclose(sv, sv2), "MPS to statevector conversion failed"
    print("Test passed: MPS to statevector and back conversion successful.")

def test_contract_sites_mps():
    """Test contract_sites produces correct shape for MPS sites"""
    sites = [np.random.rand(4,3,2)] + [np.random.rand(3,3,2)] * 1 + [np.random.rand(3,5,2)]
    c_site = contract_sites(sites) 
    assert c_site.shape == (4, 5, 2**3), f"Expected shape (4, 5, 8), got {c_site.shape}"
    print("Test passed: contract_sites for MPS produces correct shape.")

def test_contract_sites_mpdo():
    """Test contract_sites produces correct shape for MPDO sites"""
    sites = [np.random.rand(4,3,2,2)] + [np.random.rand(3,3,2,2)] * 1 + [np.random.rand(3,5,2,2)]
    c_site = contract_sites(sites)
    assert c_site.shape == (4, 5, 2**6), f"Expected shape (4, 5, 64), got {c_site.shape}"
    print("Test passed: contract_sites for MPDO produces correct shape.")

def test_decimate_mps():
    """Test decimate produces correct shape and tensors for MPS sites"""
    sites = [np.random.rand(4,3,2)] + [np.random.rand(3,3,2)] * 1 + [np.random.rand(3,5,2)]
    c_site = contract_sites(sites)
    dec_sites = decimate(c_site, 2)
    c_site2 = contract_sites(dec_sites)

    assert np.allclose(c_site, c_site2), "Decimated sites don't reconstruct original contracted site"
    assert all(s.shape == d.shape for s, d in zip(sites, dec_sites)), "Decimated site shapes don't match original"
    print("Test passed: decimate for MPS preserves information.")

def test_decimate_mpdo():
    """Test decimate produces correct shape and tensors for MPDO sites"""
    sites = [np.random.rand(4,3,2,2)] + [np.random.rand(3,3,2,2)] * 1 + [np.random.rand(3,5,2,2)]
    c_site = contract_sites(sites)
    dec_sites = decimate(c_site, 2*2)  # super local_dim is square of local_dim

    dec_sites = [s.reshape(s.shape[0], s.shape[1], 2, 2) for s in dec_sites]
    c_site2 = contract_sites(dec_sites)

    assert np.allclose(c_site, c_site2), "Decimated MPDO sites don't reconstruct original"
    assert all(s.shape == d.shape for s, d in zip(sites, dec_sites)), "Decimated MPDO shapes don't match"
    print("Test passed: decimate for MPDO preserves information.")

def test_random_mps_normalized():
    """Test random MPS produces normalized states"""
    n = 5
    mps = random_mps(n)
    norm = mps.norm()
    assert np.allclose(norm, 1.0), f"Expected norm 1.0, got {norm}"
    print("Test passed: random_mps produces normalized states.")

def test_canonize():
    """Test MPS canonization for different centers with randomized order"""
    n = 5 
    mps = random_mps(n)
    center_opts = np.arange(n)
    np.random.shuffle(center_opts)

    for center in center_opts:
        print(center)
        mps.canonize(center)
        check_center(mps, center, verbose=False)
    
    print("Test passed: MPS canonization produces correct orthogonality structure.")

def test_environments():
    """Test left and right environment computation"""
    n = 5
    mps = random_mps(n)
    norm = mps.norm()
    
    re = mps.compute_right_envs()
    le = mps.compute_left_envs()
    for i in range(n-1):
        assert np.allclose(ncon((le[i+1], re[i]), ([1,2],[1,2])), norm), f"Environment overlap at {i} doesn't match norm"
    
    print("Test passed: Left and right environments computed correctly.")

def test_environments_two_mps():
    """Test environment computation with two different MPS"""
    n = 5
    mps1 = random_mps(n)
    mps2 = random_mps(n)

    true_overlap = mps2.to_vector().conj().T @ mps1.to_vector()

    re = mps1.compute_right_envs(mps2)
    le = mps1.compute_left_envs(mps2)
    
    for i in range(n-1):
        env_overlap = ncon((le[i+1], re[i]), ([1,2],[1,2]))
        assert np.allclose(env_overlap, true_overlap), f"Environment overlap at {i} doesn't match true overlap"
    
    print("Test passed: Environments correctly compute overlap between two MPS.")

def test_overlap():
    """Test MPS overlap method"""
    n = 6
    mps1 = random_mps(n)
    mps2 = random_mps(n)
    
    mps_overlap = mps1.overlap(mps2, scaled=False)
    
    sv1 = mps1.to_vector() 
    sv2 = mps2.to_vector()
    direct_overlap = sv2.conj().T @ sv1 
    
    assert np.allclose(mps_overlap, direct_overlap), f"MPS overlap {mps_overlap} doesn't match direct overlap {direct_overlap}"
    print("Test passed: MPS overlap method computes correct value.")

def test_fast_measure():
    """Test fast_measure for statistical accuracy"""
    n = 4
    meas_sites = [1,3]
    mps = random_mps(n)

    sv = mps.to_vector()
    rdm = rdm_from_sv(sv, meas_sites)
    exact_density_vec = np.diag(rdm).real


    num_sample_opts = [100, 1000, 10000]
    for num_samples in num_sample_opts:
        mps_shots = [mps.fast_measure(meas_sites) for _ in range(num_samples)]
        mps_density = shots_to_density(mps_shots)
        mps_density_vec = shots_to_density_vec(mps_shots)
        assert np.linalg.norm(exact_density_vec - mps_density_vec, ord=2) < l2_threshold(num_samples, 2**len(meas_sites), delta=1e-2), \
                                                            f"Failed for {num_samples} samples"
    
    print("Test passed: fast_measure produces statistically accurate results.")

def test_one_rdm():
    """Test single-site reduced density matrix"""
    n = 5
    mps = random_mps(n)

    sv = mps.to_vector()
    true_rdm = rdm_from_sv(sv, [2])
    mps_rdm = mps.one_rdm(2)
    
    assert np.allclose(true_rdm, mps_rdm), "MPS one-site RDM doesn't match exact RDM"
    print("Test passed: one_rdm computes correct single-site reduced density matrix.")

def test_one_local_expectation():
    """Test single-site expectation value"""
    n = 5
    mps = random_mps(n)
    sv = mps.to_vector()
    op = random_hermitian_op(1)
    
    true_exp = np.trace(op @ rdm_from_sv(sv, [2])).real
    mps_exp = mps.one_local_expectation(2, op)
    
    assert np.allclose(true_exp, mps_exp), f"MPS expectation {mps_exp} doesn't match exact value {true_exp}"
    print("Test passed: one_local_expectation computes correct expectation value.")