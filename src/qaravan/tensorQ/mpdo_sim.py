from qaravan.core.gates import SuperOp, ID
from qaravan.core.noise import gate_time 
from .initializations import *
import copy
from tqdm.notebook import tqdm

###### nsim.py ########

##### A simpler way to do all this might be to just reshape the MPO into MPS and then apply with @ instead of ncon

def layer_to_superop(layer, span, start_idx, nm=None, name="S", time=0.0): 
    local_dim = layer[0].get_local_dim()
    
    # first we sort and pad the layer
    current_indices = set(idx for gate in layer for idx in gate.indices)
    missing_indices = set(range(span)) - current_indices
    id_gates = [ID(local_dim=local_dim, time=0.0, indices=[idx]) for idx in missing_indices]
    layer = sorted(layer + id_gates, key=lambda x: x.indices[0])
    
    # next we tensor product gates and reorder indices
    arrays = [g.to_superop(nm=nm) for g in layer]
    
    result = arrays[0]
    for i in range(1, len(arrays)):
        result = np.tensordot(result, arrays[i], axes=0)
        
    shapes = [ar.shape for ar in arrays]
    cumsum = [0] + [sum(len(s) for s in shapes[:i]) for i in range(1, len(shapes))]
    indices = [list(range(c, c + len(s))) for c, s in zip(cumsum, shapes)]
    quarters = [sum([sublist[i*len(sublist)//4:(i+1)*len(sublist)//4] for sublist in indices], []) 
                                                                                for i in range(4)]
    new_order = [item for sublist in quarters for item in sublist]
    return SuperOp(name, span, result.transpose(*new_order), start_idx=start_idx, time=time)

def noise_superop(layer, span, start_idx, nm, name="N"): 
    layer_time = max([gate_time(gate, nm) for gate in layer])
    gate_times = [0.0 for i in range(span)]
    for i in range(span): 
        for gate in layer: 
            if i in gate.indices:
                gate_times[i] = gate_time(gate, nm)

    noise_layer = [ID(nm.local_dim, max(nm.coupling*layer_time, gate_times[i]), i) 
                   for i in range(span)]
    return layer_to_superop(noise_layer, span, start_idx, nm=nm, name=name, time=np.array(gate_times))

def prod(superop1, superop2, name="S"): 
    """ returns the composition of two superoperators """
    num_indices = len(superop1.shape)
    matrix = np.tensordot(superop1.matrix, superop2.matrix, axes=(range(num_indices//2, num_indices), 
                                                                range(num_indices//2)))
    return SuperOp(name, superop1.span, matrix, start_idx=superop1.start_idx, time=superop1.time+superop2.time)

def circ_to_superop(circ, span, start_idx, nm=None): 
    new_circ = circ.build()

    superop_list = []
    for layer in new_circ.layers: 
        superop_list.append(layer_to_superop(layer, span, start_idx))
        if nm is not None: 
            noise_layer = noise_superop(layer, span, start_idx, nm)
            superop_list.append(noise_layer)

    return reduce(prod, superop_list[::-1])

def one_local(superop, site): 
    """ 
    applies single qudit superoperators; 
    superop.shape = (two left indices - top to bottom, two right indices - top to bottom)
    site.shape = (left_bond, right_bond, top_spin, bottom_spin=d)
    output.shape = (left_bond, right_bond, top_spin, bottom_spin)
    """
    output = ncon((superop, site), ([-4,-3,2,1],[-1,-2,1,2]))
    return output

def two_local(superop, site):
    """ 
    applies two qudit superoperators; 
    superop.shape = (four left indices - top to bottom, four right indices - top to bottom)
    site.shape = (left_bond, first_top_spin, first_bottom_spin, 
                    right_bond, second_top_spin, second_bottom_spin)
    output.shape = (first_top_spin, first_bottom_spin, second_top_spin, second_bottom_spin, 
                    left_bond, right_bond)
    """
    output = ncon((superop, site), ([-2,-4,-1,-3,2,4,1,3],[-5,1,2,-6,3,4])) 
    return output

def three_local(superop, site):
    """ 
    applies three qudit superoperators; 
    superop.shape = (six left indices - top to bottom, six right indices - top to bottom)
    site.shape = (first_top_spin, first_bottom_spin, second_top_spin, second_bottom_spin
                    third_top_spin, third_bottom_spin, left_bond, right_bond)
    output.shape = (first_top_spin, first_bottom_spin, second_top_spin, second_bottom_spin, 
                    third_top_spin, third_bottom_spin, left_bond, right_bond)
    """
    output = ncon((superop, site), ([-2,-4,-6,-1,-3,-5,2,4,6,1,3,5],[1,2,3,4,5,6,-7,-8])) 
    return output

def four_local(superop, site):
    """ 
    applies four qudit superoperators; 
    superop.shape = (eight left indices - top to bottom, eight right indices - top to bottom)
    site.shape = (first_top_spin, first_bottom_spin, second_top_spin, second_bottom_spin
                    third_top_spin, third_bottom_spin, fourth_top_spin, fourth_bottom_spin, left_bond, right_bond)
    output.shape = (first_top_spin, first_bottom_spin, second_top_spin, second_bottom_spin, 
                    third_top_spin, third_bottom_spin, fourth_top_spin, fourth_bottom_spin, left_bond, right_bond)
    """
    output = ncon((superop, site), ([-2,-4,-6,-8,-1,-3,-5,-7,2,4,6,8,1,3,5,7],[1,2,3,4,5,6,7,8,-9,-10])) 
    return output

def superop_action(mpdo, gate, nm=None): 
    """ applies a 1- or 2-qubit superoperator to density matrix represented as MPO """
    sites = mpdo.sites
    inds = gate.indices
    superop = gate.matrix if type(gate) == SuperOp else gate.to_superop(nm)
    
    if len(inds) == 1:
        sites[inds[0]] = one_local(superop, sites[inds[0]])
        
    elif len(inds) == 2:
        data = ncon((sites[inds[0]], sites[inds[1]]), ([-1,1,-2,-3],[1,-4,-5,-6]))
        data = two_local(superop, data)
        sites[inds[0]], sites[inds[1]] = decompose_site(data)            
    
    elif len(inds) == 3: 
        data = ncon((sites[inds[0]], sites[inds[1]], sites[inds[2]]), ([-7,1,-1,-2],[1,2,-3,-4],[2,-8,-5,-6]))
        data = three_local(superop, data)
        sites[inds[0]], sites[inds[1]], sites[inds[2]] = decompose_three_site(data)
    
    elif len(inds) == 4: 
        data = ncon((sites[inds[0]], sites[inds[1]], sites[inds[2]], sites[inds[3]]), 
                                            ([-9,1,-1,-2],[1,2,-3,-4],[2,3,-5,-6],[3,-10,-7,-8]))
        data = four_local(superop, data) 
        sites[inds[0]], sites[inds[1]], sites[inds[2]], sites[inds[3]] = decompose_four_site(data)
    
    else: 
        raise NotImplementedError(f"{len(inds)}-site gates have not been implemented") 
    
    return mpdo

def nt_sim(circ, **kwargs): 
    """ 
    noisy MPO simulation of circuit represented as gate_list 
    initial state is assumed to be the all zero state unless otherwise stated
    truncation is controlled by msvr and max_dim
    """
    init_mpdo, nm, msvr, max_dim, dm, debug, quiet = (kwargs.get('init_mpdo'), kwargs.get('nm'), kwargs.get('msvr'), 
            kwargs.get('max_dim'), kwargs.get('dm',False), kwargs.get('debug',False), kwargs.get('quiet',True))
    
    mpdo = all_zero_mpdo(circ.num_sites, circ.local_dim) if init_mpdo is None else copy.deepcopy(init_mpdo)
    circ = circ.build(nm)
    
    it = circ.layers if quiet else tqdm(circ.layers)
    for i, layer in enumerate(it): 
        for gate in layer: 
            mpdo = superop_action(mpdo, gate, nm)
            if debug: 
                pretty_print_dm(mpdo.to_matrix(), circ.local_dim)
                print(mpdo.get_skeleton())
                print("")
        
        mpdo = mpdo.compress(msvr, max_dim)
    
    if circ.meas_qubits is not None: 
        rdm = mpdo.partial_trace(circ.meas_qubits)
            
    return mpdo.to_matrix() if dm else mpdo