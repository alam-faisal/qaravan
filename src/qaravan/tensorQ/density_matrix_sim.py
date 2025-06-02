from qaravan.core.base_sim import BaseSim
from qaravan.core.utils import string_to_sv, pretty_print_dm
from qaravan.core.gates import SuperOp

import numpy as np
from ncon_torch import ncon

class DensityMatrixSim(BaseSim):
    def __init__(self, circ, init_state=None, nm=None):
        super().__init__(circ, init_state=init_state, nm=nm)    

    def initialize_state(self):
        """ 
        internal state is a rank-2n tensor with local dimension inherited from the circuit 
        init_state can be provided either as a tensor, a density matrix or a bitstring
        """
        if self.init_state is None: 
            dm = np.zeros((self.local_dim**self.num_sites, self.local_dim**self.num_sites))
            dm[0,0] = 1.0
            self.state = dm.reshape([self.local_dim]*self.num_sites*2)
        
        elif type(self.init_state) == np.ndarray:
            if len(self.init_state.shape) == 2:
                self.state = self.init_state.reshape([self.local_dim]*self.num_sites*2) 
            elif len(self.init_state.shape) == self.num_sites*2:
                self.state = self.init_state
            else:
                raise ValueError("init_state must be a rank-2 tensor or a rank-2n tensor.")

        elif type(self.init_state) == str: 
            sv = string_to_sv(self.init_state, self.circ.local_dim)
            dm = np.outer(sv, sv.conj())
            self.state = dm.reshape([self.local_dim]*self.num_sites*2)

        else:
            raise ValueError("init_state must be either a numpy array or a bitstring.") 

    def apply_gate(self, gate):
        """ not sure if this is restricted to 1- and 2-qubit gates """
        superop = gate.matrix if type(gate) == SuperOp else gate.to_superop()
        gate_indices, state_indices = generate_gate_indices(gate.indices, self.num_sites), generate_state_indices(gate.indices, self.num_sites)
        self.state = ncon((superop, self.state), (gate_indices, state_indices))

    def measure(self, meas_sites):
        """ should just use local_expectation """
        raise NotImplementedError("Measurement not yet implemented for density matrix simulator.")
    
    # TODO: allo local_ops to have span greater than 1
    def local_expectation(self, local_ops):
        """ local_ops is a list of local operators, one for each site """
        self.run(progress_bar=False)
        for i in range(self.num_sites):
            state_indices = [-(j+1) for j in range(2*self.num_sites)] 
            state_indices[i] = 1
            self.state = ncon((local_ops[i], self.state), ([-(i+1),1], state_indices))

        dm = self.state.reshape(self.local_dim**self.num_sites, self.local_dim**self.num_sites)
        return np.trace(dm).real
    
    def global_expectation(self, global_op):
        if not self.ran: 
            self.run(progress_bar=False)
        dm = self.state.reshape(self.local_dim**self.num_sites, self.local_dim**self.num_sites)
        return np.trace(dm @ global_op).real
    
    def __str__(self):
        dm = self.state.reshape(self.local_dim**self.num_sites, self.local_dim**self.num_sites)
        return pretty_print_dm(dm, self.local_dim)
    
    def get_density_matrix(self):
        return self.state.reshape(self.local_dim**self.num_sites, self.local_dim**self.num_sites)

def generate_gate_indices(locs, n): 
    numbers = [i for i in range(1,2*len(locs)+1)]
    contracted = sorted(numbers)[-len(numbers)//2:] + sorted(numbers)[:len(numbers)//2]
    
    numbers = [(int(loc)+1) for loc in locs] + [(int(loc)+n+1) for loc in locs]
    largest, smallest = sorted(numbers)[-len(numbers)//2:], sorted(numbers)[:len(numbers)//2]
    uncontracted = [-i for i in largest + smallest]
    return uncontracted + contracted

def generate_state_indices(locs, n):
    state_indices = [-i for i in range(1, 2 * n + 1)]
    pos = 1
    for i, loc in enumerate(locs):
        state_indices[loc] = pos
        state_indices[loc + n] = pos + len(locs)
        pos += 1
    return state_indices

def random_dm(n): 
    a = np.random.rand(2**n,2**n) + 1.j * np.random.rand(2**n,2**n)
    a = a@a.conj().T
    return a/np.trace(a)

def all_zero_dm(n, local_dim=2): 
    dm = np.zeros((local_dim**n, local_dim**n))
    dm[0,0] = 1.0
    return dm 

def partial_trace(dm, sites, local_dim=2): 
    system_size = int(np.log(dm.shape[0]) / np.log(local_dim))
    sites = sorted(sites)

    if len(sites) == 0: 
        return np.trace(dm).real
    
    if len(sites) == system_size:
        return dm 
    
    else: 
        dm = dm.reshape([local_dim] * system_size * 2)
        contraction = []
        counter = 0 
        for i in range(system_size):
            if i not in sites:
                counter += 1
                contraction.append(counter)

        contraction = contraction * 2 
        counter = -1
        for site in sites:
            contraction.insert(site, counter)
            counter -= 1

        for site in sites:
            contraction.insert(site + system_size, counter)
            counter -= 1

        return ncon([dm], contraction).reshape(local_dim**len(sites), local_dim**len(sites))

def expectation(dm, op): 
    return np.trace(dm @ op).real

def rdm_from_sv_full(sv, sites, local_dim=2): 
    """ Deprecated version of rdm_from_sv that uses the full state vector."""
    dm = np.outer(sv, sv.conj())
    return partial_trace(dm, sites, local_dim=local_dim)