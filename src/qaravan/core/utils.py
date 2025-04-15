import numpy as np

def endian_transform(arr):
    """ only works for qubits now """ 
    if len(arr.shape) == 1: 
        n = int(np.log2(len(arr)))
        new_order = [i for i in range(n)][::-1]
        return arr.reshape(*[2]*n).transpose(*new_order).reshape(2**n)

    elif len(arr.shape) == 2: 
        n = int(np.log2(arr.shape[0]))
        new_order = [i for i in range(n)][::-1]+[i for i in range(n,2*n)][::-1]
        return arr.reshape(*[2]*2*n).transpose(*new_order).reshape(2**n, 2**n)

    else: 
        raise ValueError("only works for 1D and 2D arrays")

def base(b,n,length):
    """ base b representation of number n assuming there are length dits """
    size = int(np.ceil(np.log(max(1,n))/np.log(b)))
    listy = [place
        for i in range(size,-1,-1)
        if (place := n//b**i%b) or i<size] or [0]
    for _ in range(length-len(listy)):
        listy.insert(0, 0)
    return listy

def generate_basis(n): 
    """ generate all possible input strings for n qubits """
    basis = []
    for i in range(2**n):
        basis.append(bin(i)[2:].zfill(n))
    return basis

def string_to_sv(string, local_dim): 
    n = len(string)
    sv = np.zeros(local_dim**n)
    sv[int(string, local_dim)] = 1.0
    return sv

def pretty_print_dm(dmat, local_dim, threshold=1e-3):
    """ prints density matrix as a mixture of quantum states """
    evals, evecs = np.linalg.eigh(dmat)
    n = dmat.shape[0]
    length = int(np.log(n) / np.log(local_dim))

    # Sort eigenvalues and eigenvectors by absolute value of eigenvalues in descending order
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    result = []
    for i, (eval, evec) in enumerate(zip(evals, evecs.T)):
        if abs(eval) > threshold:
            non_zero_indices = np.where(np.abs(evec) > threshold)[0]
            terms = []
            for idx in non_zero_indices:
                base_repr = ''.join(map(str, base(local_dim, idx, length)))
                term = f"{evec[idx]:.4f}|{base_repr}⟩"
                terms.append(term)
            state_repr = " + ".join(terms)
            result.append(f"{eval.real:.4f} * ({state_repr})")
    
    return "\n ".join(result)
    
def pretty_print_sv(sv, local_dim, threshold=1e-3):
    """ prints statevector as a linear combination of computational basis states """
    n = sv.shape[0]
    length = round(np.log(n) / np.log(local_dim))

    non_zero_indices = np.where(np.abs(sv) > threshold)[0]
    terms = []
    for idx in non_zero_indices:
        base_repr = ''.join(map(str, base(local_dim, idx, length)))
        term = f"{sv[idx]:.4f}|{base_repr}⟩"
        terms.append(term)
    state_repr = " + ".join(terms)
    
    return "".join(state_repr)
    
#################################
##### Fidelities ################
#################################

from scipy.linalg import ishermitian
def sqrtm(hmat): 
    if ishermitian(hmat, atol=1e-8):
        evals, evecs = np.linalg.eigh(hmat)
        evals[evals < 0] = 0.0
        return evecs @ np.diag(np.sqrt(evals.real)) @ evecs.conj().T
    else: 
        raise ValueError("provided matrix is not Hermitian to tolerance 1e-12")

def fidelity(rho1, rho2): 
    if len(rho1.shape) == 1:
        rho1 = np.outer(rho1, rho1.conj())
    if len(rho2.shape) == 1:
        rho2 = np.outer(rho2, rho2.conj())

    rho1_sq = sqrtm(rho1)
    arg = rho1_sq @ rho2 @ rho1_sq
    arg_sq = sqrtm(arg)
    return np.trace(arg_sq).real**2

def hellinger_fidelity(rho1, rho2):
    rho1 = np.asarray(rho1)
    rho2 = np.asarray(rho2)
    if rho1.ndim == 1:
        return np.sum(np.sqrt(rho1 * rho2))**2
    elif rho1.ndim == 2:
        return np.sum(np.sqrt(np.diag(rho1) * np.diag(rho2))).real**2
    else:
        raise ValueError("Input arrays must be either 1D or 2D.")

def two_norm_fidelity(rho1, rho2): 
    if type(rho1) == np.ndarray:
        return np.trace(rho1.conj().T @ rho2).real
    elif isinstance(rho1, MPO): 
        return (rho1.conj() @ rho2).trace(scaled=False).real
    else: 
        raise TypeError(f"provided density operator is not of valid type; must be {np.ndarray} or {MPO} not {type(rho1)}")
    
def vN_entropy(dm): 
    evals = np.linalg.eigvals(dm)
    return sum([-e*np.log(e) for e in evals if e > 0])

#=========== optimizer metadata ===========#

import sys, os, pickle, datetime
class RunContext:
    def __init__(self,
                 progress_interval=10,
                 max_iter=1000,
                 stop_ratio=1e-6,
                 checkpoint_file=None,
                 checkpoint_interval=50,
                 resume=False):
        
        self.progress_interval = progress_interval
        self.max_iter = max_iter
        self.stop_ratio = stop_ratio
        self.checkpoint_file = checkpoint_file
        self.checkpoint_interval = checkpoint_interval
        self.resume_state = self.load_checkpoint() if resume and checkpoint_file else None

    def log(self, msg):
        print(msg)
        sys.stdout.flush()

    def save_checkpoint(self, step, circ, cost_list):
        if self.checkpoint_file and step % self.checkpoint_interval == 0:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump((step, circ, cost_list), f)
            self.log(f"[Checkpoint saved at step {step}]")

    def load_checkpoint(self):
        if self.checkpoint_file and os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'rb') as f:
                step, circ, cost_list = pickle.load(f)
            self.log(f"[Resuming from checkpoint: step {step}]")
            return {"step": step, "circ": circ, "cost_list": cost_list}
        return None
    
    def step_update(self, step, circ, cost_list):
        """ handles logging, checkpointing, convergence. returns True if the run should break."""

        timestamp = datetime.datetime.now().isoformat(timespec='seconds')
        if step % self.progress_interval == 0:
            self.log(f"Step {step} at time {timestamp}: cost = {cost_list[-1]}")

        if self.checkpoint_file:
            self.save_checkpoint(step, circ, cost_list)
        
        if np.abs(cost_list[-1] - cost_list[-2]) < self.stop_ratio * cost_list[-2]:
            self.log(f"Plateau with cost {cost_list[-1]} at step {step}")
            return True

        if step == self.max_iter - 1:
            self.log(f"Max iterations reached with cost {cost_list[-1]}")
            return True

        return False