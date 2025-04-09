import numpy as np
import copy 
from scipy.linalg import block_diag, expm 
import scipy.sparse as sp
from functools import reduce
from .paulis import pauli_X, pauli_Y, pauli_Z, pauli_mapping

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
    def __init__(self, name, indices, matrix, nm=None):
        self.name = name
        self.indices = [indices] if type(indices) == int else indices
        self.span = len(self.indices)
        self.matrix = matrix
        self.nm = nm

    def __str__(self):
        return f"{self.name} gate on site(s) {self.indices}"

    def to_superop(self): 
        if self.nm is None: 
            return np.kron(self.matrix.conj(), self.matrix).reshape([self.get_local_dim()]*4*self.span)
        else: 
            return self.nm.get_superop(self.time, self.dd).reshape([self.get_local_dim()]*4*self.span)
        
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
    def __init__(self, local_dim, time, indices, nm=None, dd=False): 
        super().__init__("ID", indices, matrix=None)
        self.matrix = np.eye(local_dim**self.span)
        self.time = time
        self.dd = dd
        self.nm = nm
        
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
    def __init__(self, indices, subgate, n):
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
    
class ParamGate(Gate):
    def __init__(self, name, indices, *args): 
        super().__init__(name, indices, None) 
        
        if type(args[0]) == np.ndarray: 
            mat = args[0]
            self.update_matrix(mat)
            self.angles = self.solve_angles()
        
        else:
            self.angles = [args] if type(args) == float else args
            mat = self.construct_matrix()
            self.update_matrix(mat)
            
    def __str__(self):
        return f"{self.name} gate on site(s) {self.indices} with angle(s) {self.angles}"

    def update_matrix(self, mat): 
        self.matrix = mat
        
    def construct_matrix(self): 
        raise NotImplementedError("Subclasses must implement this method")
        
    def solve_angles(self): 
        return "unsolved" # if subclass doesn't bother implementing we probably don't need angles
    
########################################
############ QUBIT GATES ###############
########################################

def SX(indices): return Gate("SX", 
                              indices,
                              (1/np.sqrt(2)) * np.array([[1,-1j],[-1j,1]]))

def X(indices): return Gate("X", 
                            indices, 
                            np.array([[0,1],[1,0]]))

def H(indices): return Gate("H", 
                            indices, 
                            np.array([[1,1],[1,-1]])/np.sqrt(2))

def SDG(indices): return Gate("SDG", 
                              indices, 
                              np.array([[1,0],[0,-1j]]))

def CNOT(indices, n): return CNOTGate(indices, 
                                      np.array([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]]), 
                                      n)  

def CZ(indices): return Gate("CZ", 
                             indices, 
                             np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,-1]]))

def SWAP(indices): return Gate("SWAP", indices, 
                                np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]))

def iSWAP(indices): return Gate("iSWAP", indices,
                                np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]]))

class RZ(ParamGate): 
    def __init__(self, indices, *args):
        super().__init__("RZ", indices, *args)
    
    def construct_matrix(self): 
        return np.array([[1,0],[0,np.exp(1.j*self.angles[0])]])
    
class RZZ(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("RZZ", indices, *args)
    
    def construct_matrix(self): 
        return expm(-1.j*self.angles[0] * embed_operator(2, [0,1], [pauli_Z, pauli_Z], dense=True))
    
class CPhase(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("CPhase", indices, *args)
    
    def construct_matrix(self): 
        return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,np.exp(1.j*self.angles[0])]])
    
    def solve_angles(self): 
        theta = np.angle(self.matrix[3,3])
        return (theta,)

class U(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("U", indices, *args)
    
    def construct_matrix(self): 
        t,p,l = self.angles
        return np.array([[np.cos(t/2), -np.exp(1j*l)*np.sin(t/2)],
                           [np.exp(1j*p)*np.sin(t/2), np.exp(1j*(p+l))*np.cos(t/2)]])
    
    def solve_angles(self): 
        phase = np.angle(self.matrix[0,0]) 
        self.matrix = self.matrix / np.exp(1.j*phase) # removes phase for top left entry
        t = np.arccos(self.matrix[0,0])*2
        if np.abs(np.sin(t/2)) > 1e-12:
            l = np.angle(-self.matrix[0,1]/np.sin(t/2))
            p = np.angle(self.matrix[1,1]/self.matrix[0,0])-l
        else: 
            l_plus_p = np.angle(self.matrix[1,1])
            l,p = l_plus_p/2, l_plus_p/2
        
        return t,p,l        
        
    def decompose(self, basis='ZSX'): 
        if basis == 'ZSX': 
            t,p,l = self.solve_angles()#self.angles
            i = self.indices
            
            return list(reversed([
                RZ(i,p+np.pi),
                SX(i),
                RZ(i,t+np.pi),
                SX(i), 
                RZ(i,l)
            ]))   # If A = BC, we must apply C first and then B
            
        else: 
            raise NotImplementedError(f"{basis} basis decomposition has not been implemented")

class Givens(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("Givens", indices, *args)
    
    def construct_matrix(self): 
        theta = self.angles[0]
        return np.array([[1,0,0,0],
                     [0, np.cos(theta/2), -np.sin(theta/2), 0], 
                     [0, np.sin(theta/2), np.cos(theta/2), 0], 
                     [0,0,0,1]])
     
    def solve_angles(self): 
        theta = np.arccos(self.matrix[1,1])*2
        return (theta,)

class XYCoupling(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("XYCoupling", indices, *args)
    
    def construct_matrix(self): 
        theta = self.angles[0]
        xx = embed_operator(2, [0,1], [pauli_X, pauli_X], dense=True)
        yy = embed_operator(2, [0,1], [pauli_Y, pauli_Y], dense=True)
        return expm(-1.j*theta * (xx + yy))

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

class RZ01(ParamGate): 
    def __init__(self, indices, *args):
        super().__init__("RZ01", indices, *args)
    
    def construct_matrix(self): 
        return np.array([[1,0,0],[0,np.exp(1.j*self.angles[0]),0],[0,0,1]])
    
class RZ12(ParamGate): 
    def __init__(self, indices, *args):
        super().__init__("RZ12", indices, *args)
    
    def construct_matrix(self): 
        return np.array([[1,0,0],[0,1,0],[0,0,np.exp(1.j*self.angles[0])]])

class U01(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("U01", indices, *args)
    
    def construct_matrix(self): 
        t,p,l = self.angles
        return np.array([[np.cos(t/2), -np.exp(1j*l)*np.sin(t/2), 0],
                           [np.exp(1j*p)*np.sin(t/2), np.exp(1j*(p+l))*np.cos(t/2), 0],
                           [0, 0, 1]])
    
    def solve_angles(self): 
        return None # TODO implement solver 
    
    def decompose(self, basis='ZSX'): 
        if basis == 'ZSX': 
            t,p,l = self.angles
            i = self.indices
            return list(reversed([
                RZ01(i,p+np.pi),
                SX01(i),
                RZ01(i,t+np.pi),
                SX01(i), 
                RZ01(i,l)
            ]))
        else: 
            raise NotImplementedError(f"{basis} basis decomposition has not been implemented")

def random_unitary(size):
    a = np.random.rand(size, size) + 1.j * np.random.rand(size, size)
    h = a @ a.conj().T
    _, u = np.linalg.eigh(h)
    return u

def kak_unitary(params): 
    left_mat = np.kron(U(0, *params[0:3]).matrix, U(0, *params[3:6]).matrix)
    right_mat = np.kron(U(0, *params[9:12]).matrix, U(0, *params[12:15]).matrix)
    
    xx, yy, zz = np.kron(pauli_X, pauli_X), np.kron(pauli_Y, pauli_Y), np.kron(pauli_Z, pauli_Z)
    arg = sum([p*P for p,P in zip(params[6:], [xx, yy, zz])])
    center_mat = expm(1.j* arg)
    
    return left_mat @ center_mat @ right_mat

def is_unitary(u):
    return np.allclose(u @ u.conj().T, np.eye(u.shape[0]), atol=1e-10) and np.allclose(u.conj().T @ u, np.eye(u.shape[0]), atol=1e-10)