from itertools import product
import numpy as np
from qaravan.core.gates import Gate

# TODO: probably should be combined with lattices.py built on top of networkx 

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
        return edge[0] % 2 == 0 
    else: 
        assert Lx > 0, "Lx must be provided for even filter on vertical edges"
        _ , row = coords_from_index(edge[0], Lx)
        return row % 2 == 0

# TODO: add option for periodic boundaries
def nn_edges(Lx: int, Ly: int=1) -> list[tuple[int, int]]:
    """ generate nearest-neighbor edges for a 2D square lattice """ 
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

def brickwall_skeleton(Lx: int, Ly: int=1, depth: int=1) -> list[tuple[int, int]]:
    """ brickwall skeleton for 1D and 2D lattices """
    edges = nn_edges(Lx, Ly)
    skeleton = []
    
    # even horizontals, then odd horizontals
    h_edges = [e for e in edges if horiz_filter(e)]
    skeleton += [e for e in h_edges if even_filter(e, vert=False, Lx=Lx)]
    skeleton += [e for e in h_edges if not even_filter(e, vert=False, Lx=Lx)]

    # even verticals, then odd verticals
    v_edges = [e for e in edges if not horiz_filter(e)]
    skeleton += [e for e in v_edges if even_filter(e, vert=True, Lx=Lx)]
    skeleton += [e for e in v_edges if not even_filter(e, vert=True, Lx=Lx)]

    return skeleton * depth

def even_skeleton(Lx: int, Ly: int=1, depth: int=1) -> list[tuple[int, int]]:
    """ even skeleton for 1D and 2D lattices """
    edges = nn_edges(Lx, Ly)
    skeleton = []

    h_edges = [e for e in edges if horiz_filter(e)]
    skeleton += [e for e in h_edges if even_filter(e, vert=False, Lx=Lx)]

    v_edges = [e for e in edges if not horiz_filter(e)]
    skeleton += [e for e in v_edges if even_filter(e, vert=True, Lx=Lx)]

    return skeleton * depth

def odd_skeleton(Lx: int, Ly: int=1, depth: int=1) -> list[tuple[int, int]]:
    """ odd skeleton for 1D and 2D lattices """
    edges = nn_edges(Lx, Ly)
    skeleton = []

    h_edges = [e for e in edges if horiz_filter(e)]
    skeleton += [e for e in h_edges if not even_filter(e, vert=False, Lx=Lx)]

    v_edges = [e for e in edges if not horiz_filter(e)]
    skeleton += [e for e in v_edges if not even_filter(e, vert=True, Lx=Lx)]

    return skeleton * depth

def dress_skeleton(skeleton: list[tuple[int, int]], gate_type: Gate, 
            params: float | list[float] | np.ndarray | None = None) -> list[Gate]:
    """ place a gate for each edge in the skeleton 
    params can be None for unparametrized gates, 
    or a single float for same gate repeated, 
    or a list/array of floats for different gates """

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

def all_skeletons(num_gates, Lx: int, Ly: int = 1, orientation: bool=False, 
                            a2a: bool=False) -> list[list[tuple[int, int]]]:
    """ generates all possible skeletons with fixed number of gates; 
     supports nearest-neighbor or all-to-all connectivity; 
     if orientation is True, includes both directions for each edge """
    
    if not a2a: 
        pairs = nn_edges(Lx, Ly)
    else: 
        pairs = [(i,j) for i in range(Lx*Ly) for j in range(Lx*Ly) if i != j]

    if orientation: 
        pairs += [pair[::-1] for pair in pairs]

    return list(product(pairs, repeat=num_gates)) 