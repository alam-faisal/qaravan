import numpy as np
from qaravan.core import BaseSim, pauli_I, pauli_X, pauli_Y, pauli_Z
from functools import reduce
from scipy.linalg import block_diag
import re

def is_special_orthogonal(mat): 
    return bool(np.allclose(np.dot(mat.T, mat), np.eye(mat.shape[0])) and np.allclose(np.dot(mat, mat.T), np.eye(mat.shape[0])) 
            and np.isclose(np.linalg.det(mat), 1.0))

def is_special_unitary(mat): 
    return bool(np.allclose(np.dot(mat.T.conj(), mat), np.eye(mat.shape[0])) and np.allclose(np.dot(mat, mat.T.conj()), np.eye(mat.shape[0])) 
            and np.isclose(np.linalg.det(mat), 1.0))

def random_su2():
    x = np.random.randn(4)
    x = x / np.linalg.norm(x)
    a = x[0] + 1j*x[1]
    b = x[2] + 1j*x[3]
    return np.array([[a, -b.conj()],
                     [b,  a.conj()]], dtype=complex)

def matchgate_from_blocks(A,B): 
    G = np.zeros((4, 4), dtype=complex)
    G[0, 0] = A[0,0]
    G[0, 3] = A[0,1]
    G[3, 0] = A[1,0]
    G[3, 3] = A[1,1]
    G[1, 1] = B[0,0]
    G[1, 2] = B[0,1]
    G[2, 1] = B[1,0]
    G[2, 2] = B[1,1]
    return G

def random_matchgate():
    A = random_su2()
    B = random_su2()
    return matchgate_from_blocks(A, B)

def generate_majorana_ops(n):
    majorana_ops = []
    for i in range(n): 
        kron_list = [pauli_Z] * i + [pauli_X] + [pauli_I] * (n - i - 1)
        op = reduce(np.kron, kron_list)
        majorana_ops.append(op)

        kron_list = [pauli_Z] * i + [pauli_Y] + [pauli_I] * (n - i - 1)
        op = reduce(np.kron, kron_list)
        majorana_ops.append(op)

    return majorana_ops

def generate_R(n, U, indices): 
    majorana_ops = generate_majorana_ops(2)
    m_indices = np.arange(2*indices[0], 2*indices[1]+2)

    R = np.eye(2*n, dtype=complex)
    for i, ml in zip(m_indices, majorana_ops):
        for j, mr in zip(m_indices, majorana_ops):
            R[i,j] = np.trace(mr @ U @ ml @ U.conj().T)/(2**2)

    Rp = np.array([R.T[2*i,:] + 1j*R.T[2*i+1,:] for i in range(n)])
    assert is_special_orthogonal(R), "R is not special orthogonal"
    return R, Rp

def circuit_to_R(circ): 
    R = np.eye(2*circ.num_sites, dtype=complex)
    for gate in circ.gate_list:
        if gate.name == "SWAP":
            mats = [pauli_I, pauli_X, pauli_Y, pauli_Z]
            ch = np.random.choice(np.arange(4))
            mat = mats[ch]
            matrix = np.kron(mat, mat)
        elif gate.name == "M":
            matrix = gate.matrix 
        else: 
            raise ValueError(f"Unsupported gate: {gate.name}")
        
        r, _ = generate_R(circ.num_sites, matrix, gate.indices)
        R = R @ r

    Rp = np.array([R.T[2*i,:] + 1j*R.T[2*i+1,:] for i in range(circ.num_sites)])
    return R, Rp

def generate_labels(instring, outstring):
    in_labels = []
    for i, bit in enumerate(instring):
        if bit == '1':  
            in_labels.append(f'gamma_{2*i}')

    out_labels = []
    for i, bit in enumerate(outstring):
        if bit == '0':  
            out_labels.append(f'gamma_m{i}')
            out_labels.append(f'gamma_n{i}')
        elif bit == '1':
            out_labels.append(f'gamma_n{i}')
            out_labels.append(f'gamma_m{i}')

    return in_labels + out_labels + in_labels[::-1]
 
def parse_label(label):
    m = re.match(r"gamma_([nm]?)(\d+)", label)
    if not m:
        raise ValueError(f"Label '{label}' does not match expected pattern.")
    kind = m.group(1)
    if kind == "":
        kind = "gamma"  # bare gamma, use its number directly
    idx = int(m.group(2))
    return kind, idx

def get_entry(Rp, H, label_i, label_j):
    kind_i, idx_i = parse_label(label_i)
    kind_j, idx_j = parse_label(label_j)

    if kind_i == 'm' and kind_j == 'm':
        return (Rp @ H @ Rp.T)[idx_i, idx_j]  # gamma_mi, gamma_mj
    
    elif kind_i == 'm' and kind_j == 'n':
        return (Rp @ H @ Rp.conj().T)[idx_i, idx_j] # gamma_mi, gamma_nj
    
    elif kind_i == 'm' and kind_j == 'gamma': 
        return (Rp @ H)[idx_i, idx_j]
    
    elif kind_i == 'n' and kind_j == 'm':
        return (Rp.conj() @ H @ Rp.T)[idx_i, idx_j]
    
    elif kind_i == 'n' and kind_j == 'n':
        return (Rp.conj() @ H @ Rp.conj().T)[idx_i, idx_j]
    
    elif kind_i == 'n' and kind_j == 'gamma':
        return (Rp.conj() @ H)[idx_i, idx_j]
    
    elif kind_i == 'gamma' and kind_j == 'm':
        return (H @ Rp.T)[idx_i, idx_j]
    
    elif kind_i == 'gamma' and kind_j == 'n':
        return (H @ Rp.conj().T)[idx_i, idx_j]
    
    elif kind_i == 'gamma' and kind_j == 'gamma':
        return int(idx_i == idx_j)
    
    else:
        raise ValueError(f"Invalid labels: {label_i}, {label_j}")
    
def exp_mat(n, labels, Rp): 
    H = block_diag(*[np.array([[1,1j], [-1j,1]])] * n)
    mat = np.zeros((len(labels), len(labels)), dtype=complex)
    for i, label_i in enumerate(labels):
        for j, label_j in enumerate(labels):
            if i < j:
                mat[i,j] = get_entry(Rp, H, label_i, label_j)
            elif i > j:
                mat[i,j] = -mat[j,i]

    return mat

def probability(instring, outstring, circ): 
    n = len(instring) 
    num_measured_qubits = len([bit for bit in outstring if bit != 'i'])

    labels = generate_labels(instring, outstring)
    _, Rp = circuit_to_R(circ)
    mat = exp_mat(n, labels, Rp)

    norm = 2**(2*num_measured_qubits)
    return  (np.sqrt(np.linalg.det(mat))/norm).real

def sample(instring, circ, epsilon=0.0): 
    s = ''
    prev_probs = [1.0]
    for i in range(circ.num_sites):
        outstring = s + '0' + 'i'*(circ.num_sites - i - 1)   
        p = probability(instring, outstring, circ) + np.random.normal(0, epsilon)
        if np.random.rand() < p/prev_probs[-1]: 
            s += '0'  
            normalizer = p
        else: 
            s += '1'
            normalizer = prev_probs[-1] - p
        prev_probs.append(normalizer)
        
    return s 

class MatchgateSimH(BaseSim):
    def __init__(self, circ, init_state=None):
        super().__init__(circ, init_state=init_state, nm=None)

    def initialize_state(self):
        """ internal state is a SO(2n) matrix, where n is the number of sites """
        self.state = np.eye(2*self.num_sites, dtype=complex)
        self.instring = '0' * self.num_sites if self.init_state is None else self.init_state

    def generate_R(self, gate): 
        U, indices = gate.matrix, gate.indices
        majorana_ops = generate_majorana_ops(2)
        m_indices = np.arange(2*indices[0], 2*indices[1]+2)

        R = np.eye(2*self.num_sites, dtype=complex)
        for i, ml in zip(m_indices, majorana_ops):
            for j, mr in zip(m_indices, majorana_ops):
                R[i,j] = np.trace(mr @ U @ ml @ U.conj().T)/(2**2)

        assert is_special_orthogonal(R), "R is not special orthogonal"
        return R

    def apply_gate(self, gate):
        R = self.generate_R(gate)
        self.state = R @ self.state

    def probability(self, outstring): 
        n = self.num_sites
        num_measured_qubits = len([bit for bit in outstring if bit != 'i'])

        labels = generate_labels(self.instring, outstring)
        R = self.state
        Rp = np.array([R.T[2*i,:] + 1j*R.T[2*i+1,:] for i in range(self.num_sites)])
        mat = exp_mat(n, labels, Rp)

        norm = 2**(2*num_measured_qubits)
        return  (np.sqrt(np.linalg.det(mat))/norm).real
    
    def sample(self): 
        s = ''
        prev_probs = [1.0]
        for i in range(self.num_sites):
            outstring = s + '0' + 'i'*(self.num_sites - i - 1)
            p = self.probability(outstring)
            if np.random.rand() < p/prev_probs[-1]: 
                s += '0'  
                normalizer = p
            else: 
                s += '1'
                normalizer = prev_probs[-1] - p
            prev_probs.append(normalizer)
            
        return s 

class MatchgateSimS(BaseSim):
    def __init__(self, circ, init_state=None):
        super().__init__(circ, init_state=init_state, nm=None)  

    def initialize_state(self):
        """ 
        internal state is a covariance matrix, which is a 2n x 2n matrix where n is the number of sites
        init_state can be provided as a bitstring or a covariance matrix
        """
        if self.init_state is None: 
            summands = [np.array([[0,-1], [1,0]]) for i in range(self.num_sites)]
            self.state = block_diag(*summands)

        elif type(self.init_state) == str:
            mat = np.array([[0,-1], [1,0]])
            # summand will have mat or -mat on the diagonal depending on the bitstring
            summands = [-mat if b == '1' else mat for b in self.init_state]
            self.state = block_diag(*summands)

        elif type(self.init_state) == np.ndarray:
            self.state = self.init_state 

        else:
            raise ValueError("init_state must be either a covariance matrix or a bitstring.")
        
    def apply_gate(self, gate):
        gammas = [np.kron(pauli_X, pauli_I), np.kron(pauli_Y, pauli_I), np.kron(pauli_Z, pauli_X), 
                  np.kron(pauli_Z, pauli_Y)]
        rot_matrix = np.array([[0.5 * np.trace(gamma1 @ gate.matrix.conj().T @ gamma2 @ gate.matrix) 
                               for gamma2 in gammas] for gamma1 in gammas])
        summands = [np.eye(2) for i in range(self.num_sites)]
        summands[gate.indices[0]] = rot_matrix
        summands.remove(gate.indices[1])
        full_rot_matrix = block_diag(*summands)
        self.state = full_rot_matrix @ self.state @ full_rot_matrix.conj().T