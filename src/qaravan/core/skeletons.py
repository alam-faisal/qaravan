from itertools import product
import numpy as np

# TODO: probably should be combined with lattices.py


def even_skeleton(n):
    return [(i, i+1) for i in range(0, n-1, 2)]
    
def odd_skeleton(n):
    return [(i, i+1) for i in range(1, n-1, 2)]

def dressed_cnot_skeletons(num_sites, num_cnots, orientation=False, a2a=False):
    if not a2a: 
        pairs = [(i, i+1) for i in range(num_sites-1)]
    else: 
        pairs = [(i,j) for i in range(num_sites) for j in range(num_sites) if i != j]

    if orientation: 
        pairs += [pair[::-1] for pair in pairs]

    return list(product(pairs, repeat=num_cnots)) 

def brickwall_skeleton(Lx, Ly=1, depth=1):
    """ brickwall skeleton for 1D and 2D lattices """
    skeleton = []
    for y in range(Ly):
        for x in range(0, Lx-1, 2):
            skeleton.append((y*Lx + x, y*Lx + x + 1))
    for y in range(Ly):
        for x in range(1, Lx-1, 2):
            skeleton.append((y*Lx + x, y*Lx + x + 1))
    for x in range(Lx):
        for y in range(0, Ly-1, 2):
            skeleton.append((y*Lx + x, (y+1)*Lx + x))
    for x in range(Lx):
        for y in range(1, Ly-1, 2):
            skeleton.append((y*Lx + x, (y+1)*Lx + x))
    return skeleton * depth

def ss_triad_skeleton(n: int, depth: int = 1):
    """ system-system interaction in a triad contact model """
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3")
    layer1 = [(k, k+2) for k in range(0, n, 3)]
    layer2 = [(k+2, k+3) for k in range(0, n-3, 3)]
    return (layer1 + layer2) * depth

def sb_triad_skeleton(n: int, depth: int = 1):
    """ system-bath interaction in a triad contact model """
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3")
    layer1 = [(k, k+1) for k in range(0, n, 3)]
    layer2 = [(k+1, k+2) for k in range(0, n, 3)]
    return (layer1 + layer2) * depth

def triad_system_qubits(n: int):
    """ system qubits in a triad contact model """
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3")
    layer = []
    for k in range(0, n, 3):
        layer.append(k)
        layer.append(k+2)
    return layer

def triad_bath_qubits(n: int):
    """ bath qubits in a triad contact model """
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3")
    layer = []
    for k in range(0, n, 3):
        layer.append(k+1)
    return layer

def dress_skeleton(skeleton, gate_type, params=None): 
    """ place a gate for each interaction in the skeleton """
    # sometimes the same angle for all gates is desired
    if params is not None: 
        if not isinstance(params, (list, tuple, np.ndarray)):
            params = [params] * len(skeleton)

    gate_list = []
    for i, indices in enumerate(skeleton):
        if params is not None:
            gate_list.append(gate_type(indices, params[i]))
        else:
            gate_list.append(gate_type(indices)) 

    return gate_list
