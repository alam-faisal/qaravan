from .gates import Gate, SX, SX01
from .paulis import pauli_X, pauli_Y, pauli_Z
import numpy as np
import scipy
import torch

xxN, yyN, zzN = np.kron(pauli_X, pauli_X), np.kron(pauli_Y, pauli_Y), np.kron(pauli_Z, pauli_Z)
pauli_XT = torch.tensor([[0,1],[1,0]], dtype=torch.complex128)
pauli_YT = torch.tensor([[0,-1.j],[1.j,0]], dtype=torch.complex128)
pauli_ZT = torch.tensor([[1,0],[0,-1]], dtype=torch.complex128)
xxT = torch.kron(pauli_XT, pauli_XT)
yyT = torch.kron(pauli_YT, pauli_YT)
zzT = torch.kron(pauli_ZT, pauli_ZT)

def arr_constructor(x, backend="numpy"):
    if backend == "torch":
        return torch.stack([torch.stack([elem for elem in row]) for row in x]).to(dtype=torch.complex128)
    else:
        return np.array(x, dtype=np.complex128)

class ParamGate(Gate):
    def __init__(self, name, indices, *args): 
        super().__init__(name, indices, None) 
        self.backend = "torch" if torch.is_tensor(args[0]) else "numpy"
        
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
    
class RZ(ParamGate): 
    def __init__(self, indices, *args):
        super().__init__("RZ", indices, *args)
    
    def construct_matrix(self): 
        xp = np if self.backend == "numpy" else torch
        return arr_constructor([[1,0],[0,xp.exp(1.j*self.angles[0])]], backend=self.backend)
    
class RZZ(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("RZZ", indices, *args)
    
    def construct_matrix(self): 
        expm = scipy.linalg.expm if self.backend == "numpy" else torch.linalg.matrix_exp
        zz = zzN if self.backend == "numpy" else zzT
        return arr_constructor(expm(-1.j*self.angles[0] * zz))
    
class CPhase(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("CPhase", indices, *args)
    
    def construct_matrix(self): 
        xp = np if self.backend == "numpy" else torch
        return arr_constructor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,xp.exp(1.j*self.angles[0])]], 
                               backend=self.backend)
    
    def solve_angles(self): 
        theta = np.angle(self.matrix[3,3])
        return (theta,)

class U(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("U", indices, *args)
    
    def construct_matrix(self): 
        t,p,l = self.angles
        xp = np if self.backend == "numpy" else torch
        return arr_constructor([[xp.cos(t/2), -xp.exp(1j*l)*xp.sin(t/2)],
                           [xp.exp(1j*p)*xp.sin(t/2), xp.exp(1j*(p+l))*xp.cos(t/2)]], 
                           backend=self.backend)
    
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
        xp = np if self.backend == "numpy" else torch
        return arr_constructor([[1,0,0,0],
                                [0,xp.cos(theta/2),-xp.sin(theta/2),0],
                                [0,xp.sin(theta/2),xp.cos(theta/2),0],
                                [0,0,0,1]], 
                                backend=self.backend)
     
    def solve_angles(self): 
        theta = np.arccos(self.matrix[1,1])*2
        return (theta,)

class XYCoupling(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("XYCoupling", indices, *args)
    
    def construct_matrix(self): 
        theta = self.angles[0]
        expm = scipy.linalg.expm if self.backend == "numpy" else torch.linalg.matrix_exp
        xx = xxN if self.backend == "numpy" else xxT
        yy = yyN if self.backend == "numpy" else yyT
        return arr_constructor(expm(-1.j*theta * (xx + yy)), backend=self.backend)

class RZ01(ParamGate): 
    def __init__(self, indices, *args):
        super().__init__("RZ01", indices, *args)
    
    def construct_matrix(self): 
        xp = np if self.backend == "numpy" else torch
        return arr_constructor([[1,0,0],[0,xp.exp(1.j*self.angles[0]),0],[0,0,1]], 
                               backend=self.backend)
    
class RZ12(ParamGate): 
    def __init__(self, indices, *args):
        super().__init__("RZ12", indices, *args)
    
    def construct_matrix(self): 
        xp = np if self.backend == "numpy" else torch
        return arr_constructor([[1,0,0],[0,1,0],[0,0,xp.exp(1.j*self.angles[0])]], 
                               backend=self.backend)
    
class U01(ParamGate):
    def __init__(self, indices, *args):
        super().__init__("U01", indices, *args)
    
    def construct_matrix(self): 
        t,p,l = self.angles
        xp = np if self.backend == "numpy" else torch
        return arr_constructor([[xp.cos(t/2), -xp.exp(1j*l)*xp.sin(t/2), 0],
                           [xp.exp(1j*p)*xp.sin(t/2), xp.exp(1j*(p+l))*xp.cos(t/2), 0],
                           [0, 0, 1]], 
                           backend=self.backend)
    
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

def kak_unitary(params):
    backend = "torch" if torch.is_tensor(params) else "numpy"
    xp = np if backend == "numpy" else torch
    xx = xxN if backend == "numpy" else xxT
    yy = yyN if backend == "numpy" else yyT
    zz = zzN if backend == "numpy" else zzT
    expm = scipy.linalg.expm if backend == "numpy" else torch.linalg.matrix_exp

    left_mat = xp.kron(U(0, *params[0:3]).matrix, U(0, *params[3:6]).matrix)
    right_mat = xp.kron(U(0, *params[9:12]).matrix, U(0, *params[12:15]).matrix)    
    arg = sum([p*P for p,P in zip(params[6:], [xx, yy, zz])])
    center_mat = expm(1.j* arg)
    
    return left_mat @ center_mat @ right_mat