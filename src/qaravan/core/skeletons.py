from itertools import product
import numpy as np

# TODO: probably should be combined with lattices.py

def index_from_coords(x: int, y: int, Lx: int) -> int:
    """ row-major indexing with Cartesian convention """
    return y * Lx + x

def coords_from_index(idx: int, Lx: int) -> tuple[int, int]:
    """ row-major indexing with Cartesian convention """
    row, col = divmod(idx, Lx)
    return col, row

def horiz_filter(edge: tuple[int, int]) -> bool:
    """ takes a 2D edge and returns True if it is horizontal """
    return abs(edge[0] - edge[1]) == 1

def even_filter(edge: tuple[int, int], vert:bool=False, Lx: int=0) -> bool:
    """ takes a 2D edge and returns True if it is even """
    if not vert: 
        return edge[1] % 2 == 0 
    else: 
        assert Lx > 0, "Lx must be provided for even filter on vertical edges"
        _ , row = coords_from_index(edge[1], Lx)
        return row % 2 == 0

# TODO: add option for periodic boundaries
def nn_edges(Lx: int, Ly: int=1) -> list[tuple[int, int]]:
    """ generate nearest-neighbor edges for a 2D lattice """ 
    edges = []
    for row in range(Ly): 
        for col in range(Lx):
            idx = index_from_coords(col, row, Lx)

            # right neighbor
            if col < Lx - 1:
                right_idx = index_from_coords(col + 1, row, Lx)
                edges.append((idx, right_idx))
            
            # bottom neighbor
            if row < Ly - 1:
                bottom_idx = index_from_coords(col, row + 1, Lx)
                edges.append((idx, bottom_idx))

    return edges

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
