from qaravan.core.base_sim import BaseSim
from qaravan.core.utils import string_to_sv, pretty_print_sv
import numpy as np
import torch 
from ncon_torch import ncon, permute
from scipy.sparse import csc_matrix
import copy

class StatevectorSim(BaseSim):
    def __init__(self, circ, init_state=None, backend="numpy", device="cpu"):
        super().__init__(circ, init_state=init_state, nm=None)    
        self.backend = backend
        self.device = torch.device(device)

    def to_backend(self, array):
        if self.backend == "torch":
            if torch.is_tensor(array):
                return array.to(dtype=torch.complex128, device=self.device)
            else:
                return torch.tensor(array, dtype=torch.complex128, device=self.device)
        else:
            if isinstance(array, np.ndarray):
                return array
            else:
                return array.detach().cpu().numpy()
            
    def initialize_state(self):
        """ 
        internal state is a rank-n tensor with local dimension inherited from the circuit 
        init_state can be provided either as a tensor, a statevector or a bitstring
        """
        shape = [self.local_dim] * self.num_sites
        if self.init_state is None:
            sv = np.zeros(self.local_dim**self.num_sites, dtype=np.complex128)
            sv[0] = 1.0
        elif isinstance(self.init_state, str):
            sv = string_to_sv(self.init_state, self.circ.local_dim)
        elif isinstance(self.init_state, (np.ndarray, torch.Tensor)):
            sv = self.init_state
        else:
            raise ValueError("init_state must be a NumPy array, Torch tensor, or a bitstring")

        sv = sv.reshape(shape)
        self.state = self.to_backend(sv)
        
    def apply_gate(self, gate):
        mat = self.to_backend(gate.matrix)
        self.state = op_action(mat, gate.indices, self.state, local_dim=self.local_dim)

    def measure(self, meas_sites):
        raise NotImplementedError("Measurement not yet implemented for statevector simulator.")
    
    def local_expectation(self, local_ops):
        """ op is a list of 1-local Hermitian matrices """
        self.run(progress_bar=False)
        right_state = copy.deepcopy(self.state)
        for i in range(self.num_sites):
            op = self.to_backend(local_ops[i])
            state_indices = [-(j+1) for j in range(self.num_sites)] 
            state_indices[i] = 1
            right_state = ncon((op, right_state), ([-(i+1),1], state_indices))

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
    
    n = sv.ndim if sv.ndim > 1 else int(np.log(len(sv)) / np.log(local_dim))
    state = sv.reshape(*[local_dim]*n) if sv.ndim == 1 else sv
    #state = copy.deepcopy(sv).reshape(*[local_dim]*n) if sv.ndim == 1 else copy.deepcopy(sv)
    
    # locs_to_indices assumes ascending order for indices, so sort them first and transpose the operator accordingly
    sorted_indices = sorted(indices)
    sort_order = [indices.index(i) for i in sorted_indices]
    perm = sort_order + [i + len(indices) for i in sort_order]  
    op = permute(op, perm)

    gate_indices, state_indices = locs_to_indices(sorted_indices, n)
    new_sv = ncon((op, state), (gate_indices, state_indices))
    return new_sv.reshape(local_dim**n) if sv.ndim == 1 else new_sv

def all_zero_sv(num_sites, local_dim=2, dense=False, backend="numpy"):    
    if dense: 
        sv = np.zeros(local_dim**num_sites)
        sv[0] = 1.0
    else: 
        sv = csc_matrix(([1], ([0], [0])), shape=(local_dim**num_sites, 1))
    return sv

def random_sv(num_sites, local_dim=2):
    sv = np.random.rand(local_dim**num_sites) + 1j*np.random.rand(local_dim**num_sites)
    sv /= np.linalg.norm(sv)
    return sv

def partial_overlap(sv1, sv2, local_dim=2, skip=None): 
    system_size = int(np.log(len(sv1)) / np.log(local_dim))
    sites = sorted(skip) if skip is not None else []
    
    psi = sv1.reshape([local_dim] * system_size)
    phi_conj = sv2.reshape([local_dim] * system_size).conj()

    psi_labels = [0] * system_size
    phi_conj_labels = [0] * system_size

    next_contract_label = 1
    next_free_label = -1

    for i in range(system_size):
        if i in sites:
            psi_labels[i] = next_free_label
            phi_conj_labels[i] = next_free_label - len(sites)
            next_free_label -= 1
        else:
            psi_labels[i] = next_contract_label
            phi_conj_labels[i] = next_contract_label
            next_contract_label += 1
    
    contraction = ncon([psi, phi_conj], [psi_labels, phi_conj_labels])
    kept_dim = local_dim ** len(sites)
    return contraction.reshape((kept_dim, kept_dim))   

def rdm_from_sv(sv, sites, local_dim=2):
    return partial_overlap(sv, sv, local_dim=local_dim, skip=sites) 