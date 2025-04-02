from qaravan.core.base_sim import BaseSim
from qaravan.core.utils import string_to_sv, pretty_print_sv
import numpy as np
from ncon import ncon
from scipy.sparse import csc_matrix
import copy

class StatevectorSim(BaseSim):
    def __init__(self, circ, init_state=None):
        super().__init__(circ, init_state=init_state, nm=None)    

    def initialize_state(self):
        """ 
        internal state is a rank-n tensor with local dimension inherited from the circuit 
        init_state can be provided either as a tensor, a statevector or a bitstring
        """
        if self.init_state is None: 
            sv = np.zeros(self.local_dim**self.num_sites)
            sv[0] = 1.0
            self.state = sv.reshape([self.local_dim]*self.num_sites)
        
        elif type(self.init_state) == np.ndarray:
            if len(self.init_state.shape) == 1:
                self.state = self.init_state.reshape([self.local_dim]*self.num_sites) 
            elif len(self.init_state.shape) == self.num_sites:
                self.state = self.init_state
            else:
                raise ValueError("init_state must be a rank-1 tensor or a rank-n tensor.")

        elif type(self.init_state) == str: 
            sv = string_to_sv(self.init_state, self.circ.local_dim)
            self.state = sv.reshape([self.local_dim]*self.num_sites)

        else:
            raise ValueError("init_state must be either a numpy array or a bitstring.") 
        
    def apply_gate(self, gate):
        self.state = op_action(gate.matrix, gate.indices, self.state, local_dim=self.local_dim)

    def measure(self, meas_sites):
        raise NotImplementedError("Measurement not yet implemented for statevector simulator.")
    
    def local_expectation(self, local_ops):
        """ op is a list of 1-local Hermitian matrices """
        self.run(progress_bar=False)
        right_state = copy.deepcopy(self.state)
        for i in range(self.num_sites):
            state_indices = [-(j+1) for j in range(self.num_sites)] 
            state_indices[i] = 1
            right_state = ncon((local_ops[i], right_state), ([-(i+1),1], state_indices))

        return ncon((self.state.conj(), right_state), ([i for i in range(1, self.num_sites+1)], [i for i in range(1, self.num_sites+1)])).real

    def __str__(self):
        sv = self.state.reshape(self.local_dim**self.num_sites)
        return pretty_print_sv(sv, self.local_dim)

    def get_statevector(self): 
        return self.state.reshape(self.local_dim**self.num_sites)

def locs_to_indices(locs, n): 
    shifted_locs = [loc + 1 for loc in locs]
    gate_indices = [-i for i in shifted_locs] + shifted_locs

    boundaries = [0] + shifted_locs
    tensor_indices = []
    for i in range(len(shifted_locs)):
        tensor_indices += [-j for j in range(boundaries[i] + 1, boundaries[i + 1])]
        tensor_indices.append(shifted_locs[i])
    
    tensor_indices += [-j for j in range(boundaries[-1] + 1, n+1)]
    return gate_indices, tensor_indices

def op_action(op, indices, sv, local_dim=2): 
    if op.ndim != 2*len(indices): 
        op = op.reshape(*[local_dim]*2*len(indices))
    
    state = copy.deepcopy(sv).reshape(*[local_dim]*n) if sv.ndim == 1 else copy.deepcopy(sv)
    n = sv.ndim if sv.ndim > 1 else int(np.log(len(sv)) / np.log(local_dim))
    
    # locs_to_indices assumes ascending order for indices, so sort them first and transpose the operator accordingly
    sorted_indices = sorted(indices)
    sort_order = [indices.index(i) for i in sorted_indices]
    perm = sort_order + [i + len(indices) for i in sort_order]
    op = op.transpose(perm)

    gate_indices, state_indices = locs_to_indices(sorted_indices, n)
    new_sv = ncon((op, state), (gate_indices, state_indices))
    return new_sv.reshape(local_dim**n) if sv.ndim == 1 else new_sv

def all_zero_sv(num_sites, local_dim=2, dense=False):
    if dense: 
        sv = np.zeros(local_dim**num_sites)
        sv[0] = 1.0
    else: 
        sv = csc_matrix(([1], ([0], [0])), shape=(local_dim**num_sites, 1))
    return sv

def rand_sv(num_sites, local_dim=2):
    sv = np.random.rand(local_dim**num_sites) + 1j*np.random.rand(local_dim**num_sites)
    sv /= np.linalg.norm(sv)
    return sv