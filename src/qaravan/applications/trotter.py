from qaravan.core import embed_operator, pauli_Z
from qaravan.tensorQ import all_zero_dm, all_zero_sv, StatevectorSim, DensityMatrixSim, MonteCarloSim
import numpy as np
from tqdm import tqdm

def magnetization(n):
    """ return the properly normalized total magnetization operator on n qubits"""
    mz = sum(embed_operator(n, [i], [pauli_Z], dense=True) for i in range(n))
    return mz / np.sqrt(n * 2**n)

def trotter_sv_sim(ham, step_size, max_steps, op):
    """ run a statevector simulation of the Trotter circuit for a given Hamiltonian """
    circ = ham.trotter_circ(step_size, 1)
    cur_sv = all_zero_sv(ham.num_sites, dense=True)
    
    mag_list = []
    for _ in tqdm(range(max_steps)):
        sim = StatevectorSim(circ=circ, init_state=cur_sv)
        sim.run(progress_bar=False)
        cur_sv = sim.get_statevector()
        exp = np.vdot(cur_sv, op @ cur_sv).real
        mag_list.append(exp)
    return mag_list

def trotter_dm_sim(ham, step_size, max_steps, channel, op): 
    """ run a density matrix simulation of the Trotter circuit for a given Hamiltonian """
    circ = ham.trotter_circ(step_size, 1)
    cur_dm = all_zero_dm(ham.num_sites)
    
    mag_list = []
    for _ in tqdm(range(max_steps)):
        sim = DensityMatrixSim(circ=circ, nm=channel, init_state=cur_dm)
        sim.run(progress_bar=False)
        cur_dm = sim.get_density_matrix()
        noisy_exp = np.trace(op @ cur_dm).real
        mag_list.append(noisy_exp)
    return mag_list

def trotter_dm_sim_uncached(ham, step_size, max_steps, channel, op): 
    """ slow version without caching the state """
    mag_list = []
    for num_steps in tqdm(range(1, max_steps + 1)):
        circ = ham.trotter_circ(step_size, num_steps)
        sim = DensityMatrixSim(circ=circ, nm=channel)
        sim.run(progress_bar=False)
        dm = sim.get_density_matrix()
        noisy_exp = np.trace(op @ dm).real
        mag_list.append(noisy_exp)
    return mag_list

def trotter_mc_sim_uncached(ham, step_size, max_steps, channel, op, num_samples=1000): 
    """ slow version without caching the state """
    mag_list_mc = []
    for num_steps in tqdm(range(1, max_steps + 1)):
        circ = ham.trotter_circ(step_size, num_steps)
        sim = MonteCarloSim(circ=circ, nm=channel)
        mag_samples = [sim.expectation_sample(op) for _ in tqdm(range(num_samples))]
        mag_list_mc.append(np.mean(mag_samples))
    return mag_list_mc

def trotter_mc_sim(ham, step_size, max_steps, channel, op, num_samples=1000):
    """ run a Monte Carlo simulation of the Trotter circuit for a given Hamiltonian """
    circ = ham.trotter_circ(step_size, 1)
    sim = MonteCarloSim(circ=circ, nm=channel)    
    cur_sv_list = [all_zero_sv(ham.num_sites, dense=True) for _ in range(num_samples)]
    
    mag_list_mc = []
    for _ in tqdm(range(max_steps)):
        layer_samples = [sim.circuit_sample() for _ in range(num_samples)]
        mag_samples = []
        for i in range(num_samples):
            sv_sim = StatevectorSim(layer_samples[i], init_state=cur_sv_list[i])
            sv_sim.run(progress_bar=False)
            new_sv = sv_sim.get_statevector()
            cur_sv_list[i] = new_sv
            mag_samples.append(np.vdot(new_sv, op @ new_sv).real)

        mag_list_mc.append(np.mean(mag_samples))
    return mag_list_mc