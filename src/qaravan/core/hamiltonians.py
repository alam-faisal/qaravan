from .gates import *
from .circuits import Circuit
from .lattices import LinearLattice, KagomeLattice, ToricLattice
from .paulis import pauli_mapping
from scipy.linalg import expm

def terms_to_matrix(terms, num_sites, dense=True):
    """ builds full 2**n by 2**n matrix from a list of terms, where each term is a tuple (coefficient, active_sites, local_ops) """
    H = sp.csr_matrix((2**num_sites, 2**num_sites), dtype=complex)
    for coeff, active_sites, local_ops in terms:
        H += coeff * embed_operator(num_sites, active_sites, [pauli_mapping[op] for op in local_ops], factor=True)
    if dense: 
        return H.toarray()
    return H

class Hamiltonian:
    def __init__(self, coupling_types, coupling_strengths, field_types, field_strengths, lattice=None, num_x=None, locality='nn', periodic_x=False):
        """ 
        coupling_types is a list like ['xx', 'yx', 'zz'] 
        coupling_strengths is same or double the length of coupling_types
        field_types is a subset of ['x', 'y', 'z']
        field_strengths is same length as field_types
        locality is one of ['nn', 'nnn', 'a2a', 'long-range'] 
        """
        if lattice is not None:
            self.lattice = lattice
        elif num_x is not None:
            self.lattice = LinearLattice(num_x, periodic_x)
        else:
            raise ValueError("Either 'lattice' or 'num_x' must be provided.")

        self.num_sites = self.lattice.num_sites
        self.op_grouping = None
        self.coupling_types = coupling_types
        self.coupling_strengths = coupling_strengths
        self.field_types = field_types
        self.field_strengths = field_strengths
        self.locality = locality
        self.terms = []  # stores (coefficient, active_sites, local_ops)
        
        if locality == 'nn':
            for ops, strength in zip(coupling_types, coupling_strengths):
                for pairs in self.lattice.get_connections("nn"):
                    self.terms.append((strength, pairs, [ops[0], ops[1]]))

        elif locality == 'nnn':
            mid = len(coupling_strengths) // 2
            nn_strengths = coupling_strengths[:mid]
            nnn_strengths = coupling_strengths[mid:]
            nn_pairs = self.lattice.get_connections("nn")
            nnn_pairs = self.lattice.get_connections("nnn")

            for ops, nn_strength, nnn_strength in zip(coupling_types, nn_strengths, nnn_strengths):
                for pairs in nn_pairs:
                    self.terms.append((nn_strength, pairs, [ops[0], ops[1]]))
                for pairs in nnn_pairs:
                    self.terms.append((nnn_strength, pairs, [ops[0], ops[1]]))

        elif locality == 'a2a': 
            for ops, strength in zip(coupling_types, coupling_strengths):
                for pairs in self.lattice.get_connections("a2a"):
                    self.terms.append((strength, pairs, [ops[0], ops[1]]))

        elif locality == 'long-range': 
            for ops, strength in zip(coupling_types, coupling_strengths):
                for pairs in self.lattice.get_connections("a2a"):
                    self.terms.append((strength/np.abs(pairs[0]-pairs[1]), pairs, [ops[0], ops[1]]))

        for op, strength in zip(field_types, field_strengths):
            if np.abs(strength) > 1e-10:  # avoid adding zero fields
                for i in range(self.num_sites):
                    self.terms.append((strength, [i], [op]))

    def matrix(self, dense=False):
        return terms_to_matrix(self.terms, self.num_sites, dense=dense)

    def evals(self):
        return np.linalg.eigvalsh(self.matrix(dense=True)) 

    def ground(self): 
        H = self.matrix()
        return sp.linalg.eigsh(H, k=1, which='SA')       

    def propagator(self, time, dense=False):
        H = self.matrix()
        if dense:
            return sp.linalg.expm(-1j * time * H).toarray()
        return sp.linalg.expm(-1j * time * H)
    
    def evolve(self, time, init_state, dense=False):
        final_state = sp.linalg.expm_multiply(-1j * time * self.matrix(), init_state)
        if dense:
            return final_state.toarray()[:,0]
        return final_state

    def trotter_circ(self, step_size, num_steps, order=1):
        """ 
        returns Circuit object for Trotter decomposition of propagator
        currently only works if locality is nn and Hamiltonian terms can be divided into two mutually commuting groups saved in op_grouping
        so XY or Heisenberg with NNN and/or with field terms will not work
        """
        assert self.op_grouping is not None, "op_grouping is not defined or is invalid"
        gate_list = []
        if order == 1: 
            layer_list = [self.layer_from_group(group, step_size, self.num_sites) for group in self.op_grouping]
            for i in range(num_steps):
                for layer in layer_list:
                    gate_list += layer

        elif order == 2:
            l0h = self.layer_from_group(self.op_grouping[0], step_size/2, self.num_sites)
            l0 = self.layer_from_group(self.op_grouping[0], step_size, self.num_sites)
            l1 = self.layer_from_group(self.op_grouping[1], step_size, self.num_sites)
            
            gate_list += l0h
            for i in range(num_steps-1):
                gate_list += l1
                gate_list += l0

            gate_list += l1
            gate_list += l0h

        else: 
            raise NotImplementedError("Only first and second order Trotter decomposition is supported")
        
        return Circuit(gate_list, self.num_sites)

    def layer_from_group(self, group, time, num_sites):
        """ group is something like 'xx' or 'z' or 'xy' or 'even' or 'odd' """
        if group in ['even', 'odd']:
            generator_mat = sum([c*embed_operator(2, [0, 1], [pauli_mapping[t[0]], pauli_mapping[t[1]]]) for c,t in zip(self.coupling_strengths, self.coupling_types)])
        else: 
            c = self.coupling_strengths[self.coupling_types.index(group)] if len(group) == 2 else self.field_strengths[self.field_types.index(group)]
            generator_mat = c*embed_operator(len(group), [i for i in range(len(group))], [pauli_mapping[op] for op in group])

        gate_mat = expm(-1j * time * generator_mat.toarray())
        gate_list = []

        if len(group) == 1: 
            for i in range(num_sites): 
                gate = Gate('R1', [i], gate_mat)
                gate_list.append(gate)
        
        elif len(group) == 2: 
            for i in range(0, num_sites, 2):
                gate = Gate('R2', [i, i + 1], gate_mat)
                gate_list.append(gate)
            for i in range(1, num_sites-1, 2):
                gate = Gate('R2', [i, i + 1], gate_mat)
                gate_list.append(gate)

        elif group == 'even':
            for i in range(0, num_sites, 2):
                gate = Gate('R2', [i, i + 1], gate_mat)
                gate_list.append(gate)

        elif group == 'odd':
            for i in range(1, num_sites-1, 2):
                gate = Gate('R2', [i, i + 1], gate_mat)
                gate_list.append(gate)

        else: 
            raise NotImplementedError("Only field and nearest-neighbor couplings are supported")

        return gate_list
    
    def __str__(self): 
        return f"Hamiltonian with {self.num_sites} sites and {self.locality} {self.coupling_types} coupling(s) and {self.field_types} field"

class Heisenberg(Hamiltonian):
    def __init__(self, num_x, jx=1, jy=1, jz=1, h=0):
        super().__init__(
            coupling_types=['xx', 'yy', 'zz'],
            coupling_strengths=[jx, jy, jz],
            field_types=['z'],
            field_strengths=[h],
            num_x=num_x,
            locality='nn'
        )
        self.op_grouping = ['even', 'odd']

class HeisenbergNNN(Hamiltonian):
    def __init__(self, num_x, jx=1, jy=1, jz=1, jx2=1e-2, jy2=1e-2, jz2=1e-2, h=0):
        super().__init__(
            coupling_types=['xx', 'yy', 'zz'],
            coupling_strengths=[jx, jy, jz, jx2, jy2, jz2],
            field_types=['z'],
            field_strengths=[h],
            num_x=num_x,
            locality='nnn'
        )

class XY(Hamiltonian):
    def __init__(self, num_x, j=1, h=0):
        super().__init__(
            coupling_types=['xx', 'yy'],
            coupling_strengths=[j, j],
            field_types=['z'],
            field_strengths=[h],
            num_x=num_x,
            locality='nn'
        )
        self.op_grouping = ['even', 'odd']

class XYNNN(Hamiltonian):
    def __init__(self, num_x, j=1, j2=0.01, h=0):
        super().__init__(
            coupling_types=['xx', 'yy'],
            coupling_strengths=[j, j, j2, j2],
            field_types=['z'],
            field_strengths=[h],
            num_x=num_x,
            locality='nnn'
        )

    def layer_from_group(self, group, time, num_sites):
        if group in ['even', 'odd']:
            generator_mat = sum(
                c * embed_operator(2, [0, 1], [pauli_mapping[t[0]], pauli_mapping[t[1]]])
                for c, t in zip(self.coupling_strengths, self.coupling_types)
            )
        else:
            generator_mat = sum(
                c * embed_operator(3, [0, 1, 2], [pauli_mapping[t[0]], pauli_mapping[t[1]], pauli_mapping[t[2]]])
                for c, t in zip(self.coupling_strengths, self.coupling_types)
            )

class TFI(Hamiltonian):
    def __init__(self, num_x, jz=1, h=0, periodic_x=False):
        super().__init__(
            coupling_types=['zz'],
            coupling_strengths=[jz],
            field_types=['x'],
            field_strengths=[h],
            num_x=num_x,
            locality='nn', 
            periodic_x=periodic_x
        )
        self.op_grouping = ['x', 'zz']

class LMG(Hamiltonian):
    def __init__(self, num_x, jx=1, h=0):
        super().__init__(
            coupling_types=['xx'],
            coupling_strengths=[jx],
            field_types=['z'],
            field_strengths=[h],
            num_x=num_x,
            locality='a2a'
        )

class ChiralHeisenberg(Hamiltonian):
    def __init__(self, row_layout, jx=1, jy=1, jz=1, chiral_strength=0.01, h=0, sp=None):
        lattice = KagomeLattice(row_layout)
        if sp is not None:
            lattice.spruce(sp)
        super().__init__(coupling_types=['xx', 'yy', 'zz'],
                         coupling_strengths=[jx, jy, jz],
                         field_types=['z'],
                         field_strengths=[h],
                         lattice=lattice,
                         locality='nn')
        
        self.chiral_strength = chiral_strength
        if np.abs(chiral_strength) > 1e-10:
            for triplet in self.lattice.triangle_terms:
                self.terms.extend(self.chiral_terms(triplet))
       
    def pairwise_terms(self, pairs):
        return [(j, p, op) for p in pairs for (j,op) in zip(self.coupling_strengths, self.coupling_types)]

    def chiral_terms(self, triplet):
        ops = ['xyz', 'yzx', 'zxy', 'xzy', 'zyx', 'yxz']
        return [(self.chiral_strength if op in ['xyz', 'yzx', 'zxy'] else -self.chiral_strength, triplet, op) for op in ops]

    def primitive_matrices(self):
        """ returns small 8 by 8 matrix for triangle primitives and 4 by 4 matrix for edge primitives """
        triangle_terms = self.pairwise_terms([(0,1), (1,2), (0,2)]) + self.chiral_terms((0,1,2))
        tmat = sum([coeff * embed_operator(3, sites, [pauli_mapping[op] for op in local_ops], factor=True) for coeff, sites, local_ops in triangle_terms])

        edge_terms = self.pairwise_terms([(0,1),])
        emat = sum([coeff * embed_operator(2, sites, [pauli_mapping[op] for op in local_ops], factor=True) for coeff, sites, local_ops in edge_terms])
        return tmat.toarray(), emat.toarray()

    def grouped_terms(self): 
        """ returns a list of (indices, mat) where mat acts on qubits at indices, not mat is either 4 by 4 or 8 by 8 and each index triplet or pair only appears once 
        used by ham_action function; output cannot be used to compute the full Hamiltonian matrix since indices are possibly non-contiguous """
        tmat, emat = self.primitive_matrices()
        triangle_edges = [set(pair) for t in self.lattice.triangle_terms for pair in [(t[0], t[1]), (t[1], t[2]), (t[0], t[2])]]
        edge_terms = [pair for pair in self.lattice.nn_pairs if set(pair) not in triangle_edges]
        return [(indices, tmat) for indices in self.lattice.triangle_terms] + [(indices, emat) for indices in edge_terms]
            
class ToricCode(Hamiltonian):
    def __init__(self, Lx, Ly, hx=1.0, hz=1.0):
        self.lattice = ToricLattice(Lx, Ly)
        self.num_sites = self.lattice.num_sites
        self.terms = []

        for v_edges in self.lattice.vertex_terms:
            self.terms.append((-hx, v_edges, ['x'] * 4))

        for p_edges in self.lattice.plaquette_terms:
            self.terms.append((-hz, p_edges, ['z'] * 4))

    def __str__(self):
        return (f"Toric Code Hamiltonian on {self.lattice.Lx}×{self.lattice.Ly} "
                f"lattice with {self.num_sites} qubits")