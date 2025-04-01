import numpy as np
from .gates import *
from .paulis import pauli_commute, pauli_mapping, pauli_multiply, pauli_strings, random_pauli_string

class Noise:
    def get_kraus(self, *args, **kwargs):
        """ returns list of Kraus operators of the noise channel."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_superop(self, *args, **kwargs):
        """ returns the superoperator representation of the noise."""
        kraus_list = self.get_kraus(*args, **kwargs)
        return sum([np.kron(k.conj(), k) for k in kraus_list])

class ThermalNoise(Noise): 
    def __init__(self, t1, t2, one_site_time, two_site_time, local_dim, coupling=1.0):
        self.t1 = t1
        self.t2 = t2
        self.idling_t2 = coupling*t2 + (1-coupling)*t1*2
        self.one_site_time = one_site_time
        self.two_site_time = two_site_time
        self.local_dim = local_dim
        self.coupling = coupling
    
class QubitNoise(ThermalNoise): 
    def __init__(self, t1, t2, one_qubit_time, two_qubit_time, coupling=1.0):
        super().__init__(t1, t2, one_qubit_time, two_qubit_time, 2, coupling)
        
    def get_kraus(self, time, dd=False): 
        t1 = self.t1
        t2 = self.t2 if not dd else self.idling_t2
        p1 = 1-np.exp(-time/t1)
        p2 = 1-np.exp(-time/t2)
        p1c = 1-p1
        p2c = 1-p2
        
        kraus_list = [np.zeros((2,2)) for _ in range(4)]
        
        kraus_list[0][0,0] = np.sqrt(p2)
        kraus_list[0][1,1] = -np.sqrt(p2*p1c)
        
        kraus_list[1][0,1] = -np.sqrt(p1*p2)
        
        kraus_list[2][0,0] = np.sqrt(p2c)
        kraus_list[2][1,1] = np.sqrt(p1c*p2c)
        
        kraus_list[3][0,1] = np.sqrt(p1*p2c)
        
        return kraus_list
        
class QutritNoise(ThermalNoise): 
    def __init__(self, t1, t2, one_qutrit_time, two_qutrit_time, coupling=1.0):
        super().__init__(t1, t2, one_qutrit_time, two_qutrit_time, 3, coupling)
    
    def get_kraus(self, time, dd=False): 
        t1 = self.t1
        t2 = self.t2 if not dd else self.idling_t2
        tr = t1*t2/(t1+t2)
        p1 = 1-np.exp(-time/t1)
        p2 = 1-np.exp(-time/t2)
        pr = 1-np.exp(-time/tr)
        p1c = 1-p1
        p2c = 1-p2
        prc = 1-pr

        kraus_list = [np.zeros((3,3)) for _ in range(9)]

        kraus_list[0][0,0] = np.sqrt(p2c)
        kraus_list[0][1,1] = np.sqrt(prc)
        kraus_list[0][2,2] = np.sqrt(prc)

        kraus_list[1][0,1] = np.sqrt(p2c*p1)
        kraus_list[2][1,2] = np.sqrt(p2c*p1)

        kraus_list[3][0,0] = np.sqrt(p2) * (1/np.sqrt(2))
        kraus_list[3][1,1] = -np.sqrt(p1c*p2) * (1/np.sqrt(2))
        kraus_list[3][2,2] = np.sqrt(p1c*p2) * (1/np.sqrt(2))

        kraus_list[4][0,1] = np.sqrt(p1*p2) * (1/np.sqrt(2))
        kraus_list[5][1,2] = -np.sqrt(p1*p2) * (1/np.sqrt(2))

        kraus_list[6][0,0] = np.sqrt(p2) * (1/np.sqrt(2))
        kraus_list[6][1,1] = np.sqrt(p1c*p2) * (1/np.sqrt(2))
        kraus_list[6][2,2] = -np.sqrt(p1c*p2) * (1/np.sqrt(2))

        kraus_list[7][0,1] = np.sqrt(p1*p2) * (1/np.sqrt(2))
        kraus_list[8][1,2] = np.sqrt(p1*p2) * (1/np.sqrt(2))
        return kraus_list        
    
class PauliNoise(Noise):
    def __init__(self, strings, probs):
        assert np.abs(np.sum(probs) - 1) < 1e-6, "Probabilities must sum to 1."
        self.probs = [p for p in probs if np.abs(p) > 1e-6]
        self.strings = [s for s, p in zip(strings, probs) if np.abs(p) > 1e-6]
        self.num_sites = len(strings[0])
        self.local_dim = 2
        
    def sample_string(self):
        return self.strings[np.random.choice(len(self.strings), p=self.probs)]
    
    def symp_matrix(self, full=False): 
        """ symplectic overlap matrix for the terms of the channel."""
        if not full: 
            return np.array([[(-1)**(not pauli_commute(s1, s2)) for s1 in self.strings] for s2 in self.strings])
        else: 
            strings = pauli_strings(self.num_sites)
            return np.array([[(-1)**(not pauli_commute(s1, s2)) for s1 in strings] for s2 in strings])
            
    def ptm(self): 
        strings = pauli_strings(self.num_sites)
        fidelities = []
        for s in strings: 
            fidelity = 0.0
            for s2, p in zip(self.strings, self.probs): 
                if pauli_commute(s, s2): 
                    fidelity += p
                else:
                    fidelity -= p
            fidelities.append(fidelity)
        return np.diag(fidelities)
    
    def fidelities(self):
        """ returns pauli fidelities w.r.t. own strings. """
        return self.symp_matrix() @ self.probs 
    
    def get_kraus(self, *args, **kwargs):
        ops = [embed_operator(len(s), [i for i in range(len(s))], [pauli_mapping[op] for op in s], dense=True) for s in self.strings]
        return [np.emath.sqrt(p)*op for p, op in zip(self.probs, ops)]
        #return [np.sign(p)*np.sqrt(np.abs(p))*op for p, op in zip(self.probs, ops)]
    
    def pinv(self): 
        """ returns the pseudoinverse of the channel. """
        inv_fidelities = 1/np.diag(self.ptm())
        inv_probs = np.linalg.inv(self.symp_matrix(full=True)) @ inv_fidelities
        return PauliNoise(pauli_strings(self.num_sites), inv_probs)
    
    def compose(self, other):
        """ compose two noise channels. """
        new_strings = []
        new_probs = []

        for s1, p1 in zip(self.strings, self.probs):
            for s2, p2 in zip(other.strings, other.probs):
                new_string, phase = pauli_multiply(s1, s2)
                new_strings.append(new_string)
                new_probs.append(p1 * p2)

        unique_strings = list(set(new_strings))
        new_probs_dict = {s: 0 for s in unique_strings}
        for s, p in zip(new_strings, new_probs):
            new_probs_dict[s] += p
        new_probs = [new_probs_dict[s] for s in unique_strings]

        return PauliNoise(unique_strings, new_probs)
    
    def __str__(self): 
        noise_dict = {str(s): np.around(p, 3) for s, p in zip(self.strings, self.probs)}
        return str(noise_dict)

def gate_time(gate, nm):   
    if gate.name[0:4] == "CNOT": 
        return nm.two_site_time
    elif gate.name[0:2] == "RZ" or gate.name[0:2] == "CU":
        return 0.0
    else: 
        return nm.one_site_time
    
def random_pauli_channel(n,n_strings=5,c=0.1):
    strings = ['i'*n] + [random_pauli_string(n) for _ in range(n_strings-1)]
    probs = [1,] + [c*np.random.rand() for _ in range(n_strings-1)] 
    probs /= np.sum(probs)
    return PauliNoise(strings, probs)