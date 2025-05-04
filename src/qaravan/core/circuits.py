from .gates import ID, SuperOp, random_unitary, Gate
from .param_gates import kak_unitary
from .noise import gate_time, ThermalNoise
import numpy as np
import copy
from functools import reduce 

class Circuit:
    def __init__(self, gate_list, n=None, local_dim=None, meas_sites=None):
        self.gate_list = gate_list

        if len(gate_list) != 0: 
            sites = np.concatenate([gate.indices for gate in gate_list])
            self.num_sites = int(max(sites) + 1) if n is None else n
            self.local_dim = gate_list[0].get_local_dim()

        else: 
            self.num_sites = n 
            self.local_dim = local_dim

        self.layers = None
        self.built = False
        self.meas_sites = meas_sites
        
    def update_gate(self, gate_idx, new_array): 
        self.gate_list[gate_idx].matrix = new_array
        if self.layers is not None:
            self.construct_layers()
        
    def decompose(self, basis='ZSX'): 
        gate_list = []
        for gate in self.gate_list:
            if gate.name == "U" and gate.angles is not None:
                decomposed_gates = gate.decompose(basis)
                gate_list.extend(decomposed_gates)
            else:
                gate_list.append(gate)

        self.gate_list = gate_list
        
    def construct_layers(self): 
        layers = []
        for gate in self.gate_list:
            place_available = True
            layer_ind = len(layers) - 1

            # first we find the oldest layer that fails to commute with current gate
            while place_available and layer_ind >= 0:
                g_indices = set(gate.indices)
                existing_indices = {ind for gate in layers[layer_ind] for ind in gate.indices}
                if g_indices.intersection(existing_indices): 
                    place_available = False
                else: 
                    layer_ind -= 1

            # then we add current gate to the following layer
            if layer_ind < len(layers) - 1: 
                layers[layer_ind + 1].append(gate)
            else: 
                layers.append([gate])

        self.layers = layers
    
    def add_noise(self, nm): 
        layers = []
        for layer in self.layers: 
            if not any(isinstance(obj, SuperOp) for obj in layer):
                if isinstance(nm, ThermalNoise): 
                    layer_time = max([gate_time(gate, nm) for gate in layer])
                    active_qubits = set(np.concatenate([gate.indices for gate in layer]))
                    idle_qubits = set(range(self.num_sites)) - active_qubits
                    noise_layer = [ID(nm.local_dim, layer_time, i, nm=nm, dd=(i in idle_qubits))
                                    for i in range(self.num_sites)] 
                    
                else: 
                    noise_layer = [ID(2, 0.0, [i for i in range(self.num_sites)], nm=nm)] 
                
                layer.extend(noise_layer)
                layers.append(layer)
            else: 
                layers.append(layer)
        
        self.layers = layers
        
    def build(self, nm=None):
        """ this changes the current circuit instance and returns it """
        if not self.built:
            self.decompose()
            self.construct_layers()
            if nm is not None: 
                self.add_noise(nm)
            self.built = True
        return self
    
    def old_build(self, nm=None): 
        """ DEPRECATED: note that this returns a new circuit instance built with provided noise model
        and leaves the current circuit instance unchanged """
        new_circ = copy.deepcopy(self)
        new_circ.decompose()
        new_circ.construct_layers()
        if nm is not None: 
            new_circ.add_noise(nm)
        
        new_circ.built = True    
        return new_circ
    
    def add_meas(self, meas_sites): 
        self.meas_sites = meas_sites
        
    def dag(self): 
        gate_list = [g.dag() for g in self.gate_list[::-1]]
        return Circuit(gate_list, n=self.num_sites, local_dim=self.local_dim)
        
    def to_matrix(self): 
        return circ_to_mat(self)
        
    def __str__(self): 
        if self.layers is not None: 
            full_str = []
            for layer in self.layers:
                layer_str = " \n".join([str(gate) for gate in layer])
                full_str.append(layer_str)
            full_str = " \n\n".join(full_str)
            return full_str
        
        else: 
            return " \n".join([str(gate) for gate in self.gate_list])  

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.gate_list[key]
        elif isinstance(key, slice):
            return Circuit(self.gate_list[key], n=self.num_sites, local_dim=self.local_dim)
        else:
            raise TypeError("Invalid index type.")
        
    def __len__(self):
        return len(self.gate_list)
    
    def __add__(self, other):
        return compose_circuits([self, other])
        
    def draw(self): # TODO 
        """ use svgwrite to visualize the circuit """
        return None

def compose_circuits(circ_list): 
    gate_list = [circ.gate_list for circ in circ_list]
    gate_list = [gate for sublist in gate_list for gate in sublist]
    return Circuit(gate_list, n=circ_list[0].num_sites)

def circ_to_mat(circ, n=None): 
    if circ.layers == None: 
        circ.construct_layers()
    n = circ.num_sites if n is None else n
    
    layer_mats = []    
    for layer in circ.layers: 
        missing_indices = set(range(n)) - set(np.concatenate([gate.indices for gate in layer]))
        for index in missing_indices: 
            layer.append(ID(circ.local_dim, 0.0, [index]))
        
        sorted_layer = sorted(layer, key=lambda gate: gate.indices[0])
        mat_list = [gate.matrix for gate in sorted_layer]
        layer_mats.append(reduce(np.kron, mat_list))
        
    return np.linalg.multi_dot(layer_mats[::-1]) if len(layer_mats) > 1 else layer_mats[0]

def shift_gate_list(circ, start_site):
    """ return a new gate list with all gate indices shifted by start_site."""
    def _shifted_gate(gate, start_site):
        new_gate = copy.deepcopy(gate)
        new_gate.indices = np.array(gate.indices) + start_site
        return new_gate
    
    return [
        _shifted_gate(gate, start_site)
        for gate in circ.gate_list
    ]

def two_local_circ(skeleton, params=None, mag=None):
    gate_list = []
    for i,indices in enumerate(skeleton):
        if mag is None and params is None:
            mat = random_unitary(4)
        elif params is not None:
            mat = kak_unitary(params[15*i:15*(i+1)])
        else:
            mat = kak_unitary(np.random.rand(15) * mag)

        gate_list.append(Gate("rand_U", indices, mat))
    return Circuit(gate_list)