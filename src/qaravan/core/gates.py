import numpy as np
import copy 
from scipy.linalg import block_diag
import scipy.sparse as sp
from functools import reduce
from .paulis import pauli_mapping, pauli_Y, pauli_Z

def embed_operator(num_sites, active_sites, local_ops, local_dim=2, dense=False, factor=False):
    # only works if Pauli string is provided 
    if type(local_ops) == str: 
        local_ops = [pauli_mapping[op] for op in local_ops]

    # only works if active_sites is contiguous
    if type(local_ops) == np.ndarray:  
        full_op = [np.eye(local_dim, dtype=complex) for i in range(num_sites) if i not in active_sites]
        full_op.insert(active_sites[0], local_ops)
        return reduce(np.kron, full_op)
        
    # if local_ops is a list of 1-qubit matrices 
    if dense: 
        full_op = [np.eye(local_dim, dtype=complex) for _ in range(num_sites)]
        for site, op in zip(active_sites, local_ops):
            full_op[site] = op if not factor else op/2
        return reduce(np.kron, full_op)
    
    else: 
        full_op = [sp.eye(local_dim, format='csr', dtype=complex) for _ in range(num_sites)]
        for site, op in zip(active_sites, local_ops):
            full_op[site] = sp.csr_matrix(op) if not factor else sp.csr_matrix(op/2)
        return reduce(sp.kron, full_op)

class Gate:
    def __init__(self, name, indices, matrix, time=None):
        self.name = name
        self.indices = [indices] if type(indices) == int else indices
        self.span = len(self.indices)
        self.matrix = matrix
        self.time = time

    def __str__(self):
        return f"{self.name} gate on site(s) {self.indices}"

    def to_superop(self): 
        return np.kron(self.matrix.conj(), self.matrix).reshape([self.get_local_dim()]*4*self.span)
        
    def get_local_dim(self):
        return int(self.matrix.shape[0]**(1/self.span))
    
    def dag(self): 
        g = copy.deepcopy(self)
        g.matrix = g.matrix.conj().T
        return g    
    
class SuperOp:
    """ deprecated; need to be debugged first """
    def __init__(self, name, span, matrix, start_idx=0, time=0.0):
        self.name = name
        
        self.start_idx = start_idx
        self.indices = [start_idx+i for i in range(span)]
        self.span = span
        
        self.matrix = matrix
        self.shape = matrix.shape 
        self.time = time 
        
    def shift(self, new_start_idx): 
        return SuperOp(self.name, self.span, self.matrix, start_idx=new_start_idx, time=self.time)
    
    def __str__(self):
        return f"{self.name} superoperator on site(s) {self.indices}"

###########################################
############# SUBCLASSES ##################
###########################################

class ID(Gate): 
    def __init__(self, local_dim, time, indices, nm, dd=False): 
        super().__init__("ID", indices, matrix=None, time=time)
        self.matrix = np.eye(local_dim**self.span)
        self.dd = dd
        self.nm = nm

    def to_superop(self): 
        return self.nm.get_superop(self.time, self.dd).reshape([self.get_local_dim()] * 4 * self.span)
        
    def __str__(self):
        str = f"Idling on site(s) {self.indices} with {type(self.nm).__name__} for time {self.time}"
        if self.dd: 
            str += "; dynamical decoupling"
        return str
    
# n is needed for PBC where *non-local* CUs are allowed across the boundary
# indices = (targets, control)
# note target indices HAVE to be in ascending order and nearest neighbors
# applies subgate when control wire is at 1 and applies identity otherwise

class CUGate(Gate):
    def __init__(self, indices, subgate):
        d = subgate.get_local_dim()
        submat = subgate.matrix
        s = submat.shape[0]
        indices = np.array([int(i) for i in indices])
        r = np.min(np.abs(indices[:-1] - indices[-1]))
        
        if r > 1: # 'long'-range gate across boundary
            raise NotImplementedError("Long range CUGate has not been implemented yet")
            
        else: # nearest neighbor gate 
            mat_list = [np.eye(s), submat] + [np.eye(s)] * (d-2)
            mat = block_diag(*mat_list)
            
            if indices[-2] < indices[-1]: 
                indices = indices
                mat = mat.reshape(d,s,d,s).transpose(1,0,3,2).reshape(d*s,d*s)
                bottom_heavy = False
            elif indices[-1] < indices[0]: 
                indices = np.concatenate(([indices[-1]], indices[0:-1]))
                bottom_heavy = True
            else: 
                raise ValueError("indices does not have allowed ordering")
        
        name = f"CU{d}"
        super().__init__(name, indices, mat)
        self.bottom_heavy = bottom_heavy
        
    def __str__(self):
        title = "bottom heavy" if self.bottom_heavy else "top heavy"
        return f"{title} {self.name} gate on site(s) {self.indices}"
    
# n is needed for PBC where *non-local* CNOTs are allowed across the boundary
# indices = (target, control)
class CNOTGate(Gate):
    """ should become subclass of CUGate """
    def __init__(self, indices, matrix, n):
        d = int(np.sqrt(matrix.shape[0]))
        indices = [int(i) for i in indices]
        
        if np.abs(indices[0] - indices[1]) > 1: # 'long'-range gate across boundary
            temp = np.array(indices) + 1
            if temp[0]%n < temp[1]%n: 
                indices = indices
                mat = matrix
                bottom_heavy = False
            else: 
                indices = indices[::-1]
                mat = matrix.reshape(d,d,d,d).transpose(1,0,3,2).reshape(d*d,d*d)
                bottom_heavy = True

        else: # nearest neighbor gate 
            if indices[0] < indices[1]: 
                indices = indices
                mat = matrix
                bottom_heavy = False
            else: 
                indices = indices[::-1]
                mat = matrix.reshape(d,d,d,d).transpose(1,0,3,2).reshape(d*d,d*d)
                bottom_heavy = True
        
        name = f"CNOT{d}"
        super().__init__(name, indices, mat)
        self.bottom_heavy = bottom_heavy
        
    def __str__(self):
        title = "bottom heavy" if self.bottom_heavy else "top heavy"
        return f"{title} {self.name} gate on site(s) {self.indices}"
        
########################################
############ QUBIT GATES ###############
########################################

def SX(indices): return Gate("SX", 
                              indices,
                              (1/np.sqrt(2)) * np.array([[1,-1j],[-1j,1]]))

def X(indices): return Gate("X", 
                            indices, 
                            np.array([[0,1],[1,0]]))

def Y(indices): return Gate("Y",
                            indices,
                            pauli_Y)

def Z(indices): return Gate("Z",
                            indices,
                            pauli_Z)

def H(indices): return Gate("H", 
                            indices, 
                            np.array([[1,1],[1,-1]])/np.sqrt(2))

def SDG(indices): return Gate("SDG", 
                              indices, 
                              np.array([[1,0],[0,-1j]]))

def CNOT(indices, n=1000): return CNOTGate(indices, 
                                      np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]), 
                                      n=n)  

def CZ(indices): return Gate("CZ", 
                             indices, 
                             np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]))

def SWAP(indices): return Gate("SWAP", indices, 
                                np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]))

def iSWAP(indices): return Gate("iSWAP", indices,
                                np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]))

#########################################
############# QUTRIT GATES ##############
#########################################

def X01(indices): return Gate("X01", 
                              indices, 
                              np.array([[0,-1j,0],[-1j,0,0],[0,0,1]]))

def SX01(indices): return Gate("SX01", 
                               indices, 
                               (1/np.sqrt(2)) * np.array([[1,-1j,0],[-1j,1,0],[0,0,np.sqrt(2)]]))

def X12(indices): return Gate("X12", 
                              indices, 
                              np.array([[1,0,0],[0,0,-1j],[0,-1j,0]]))

def SX12(indices): return Gate("SX12", 
                               indices, 
                               (1/np.sqrt(2)) * np.array([[np.sqrt(2),0,0],[0,1,-1j],[0,-1j,1]]))

def H01(indices): return Gate("H01", 
                              indices, 
                              np.array([[1/np.sqrt(2),1/np.sqrt(2),0],[1/np.sqrt(2),-1/np.sqrt(2),0],[0,0,1]]))

def SDG01(indices): return Gate("SDG01", 
                              indices, 
                              np.array([[1,0,0],[0,-1j,0],[0,0,1]]))

def CNOT3(indices, n): return CNOTGate(indices, 
                                      np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 1, 0., 0., 0., 0.],
                                       [0., 0., 1/np.sqrt(2), 0., 0., 1/np.sqrt(2), 0., 0., 0.],
                                       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                       [0., 0.,  -1/np.sqrt(2), 0., 0., 1/np.sqrt(2), 0., 0., 0.],
                                       [0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                       [0., 0., 0., 0., 0., 0., 0., 1j, 0.],
                                       [0., 0., 0., 0., 0., 0., 0., 0., 1.]]).T, 
                                      n)   

def SWAP3(indices): return Gate("SWAP3", indices, 
                                np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                       [0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                       [0., 0., 0, 0., 0., 0, 1., 0., 0.],
                                       [0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                       [0., 0.,  0, 0., 0., 0, 0., 1., 0.],
                                       [0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                       [0., 0., 0., 0., 0., 1., 0., 0, 0.],
                                       [0., 0., 0., 0., 0., 0., 0., 0., 1.]]))

def random_unitary(size):
    a = np.random.rand(size, size) + 1.j * np.random.rand(size, size)
    h = a @ a.conj().T
    _, u = np.linalg.eigh(h)
    return u

def is_unitary(u):
    return np.allclose(u @ u.conj().T, np.eye(u.shape[0]), atol=1e-10) and np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=1e-10)

pauli_gate_map = {
    'x': X,
    'y': Y,
    'z': Z
}

def pauli_string_to_gates(string):
    return [pauli_gate_map[op](i) for i, op in enumerate(string) if op != 'i']