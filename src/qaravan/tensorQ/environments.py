from qaravan.tensorQ.statevector_sim import StatevectorSim, partial_overlap, all_zero_sv, op_action
from qaravan.core.utils import RunContext 
from qaravan.core.circuits import two_local_circ
from tqdm import tqdm
import numpy as np

def sv_environment(circ, left_sv, gate_idx): 
    mat, indices = circ[gate_idx].matrix, circ[gate_idx].indices

    circ1 = circ[:gate_idx]
    circ2 = circ[gate_idx+1:]

    sim1 = StatevectorSim(circ1, init_state=None)
    sim2 = StatevectorSim(circ2.dag(), init_state=left_sv)

    sv1 = sim1.run(progress_bar=None).reshape(2**circ.num_sites)
    sv2 = sim2.run(progress_bar=None).reshape(2**circ.num_sites)

    return partial_overlap(sv1, sv2, skip=indices), mat

def cache_states(circ, left_sv):
    pre_states = [all_zero_sv(circ.num_sites, dense=True)]  
    state = pre_states[0]
    for gate in tqdm(circ.gate_list, desc='Pre states'):
        state = op_action(gate.matrix, gate.indices, state)
        pre_states.append(state)

    post_states = [left_sv]
    state = left_sv
    for gate in tqdm(reversed(circ.gate_list), desc='Post states'):
        state = op_action(gate.matrix.conj().T, gate.indices, state)
        post_states.append(state)
    post_states = list(reversed(post_states))
    return pre_states[:-1], post_states[1:]

def environment_update(circ, gate_idx, pre_states, post_states, direction='right'):
    indices = circ[gate_idx].indices
    env = partial_overlap(pre_states[gate_idx], post_states[gate_idx], skip=indices)
    x,s,yh = np.linalg.svd(env)
    new_mat = yh.conj().T @ x.conj().T
    circ.update_gate(gate_idx, new_mat)

    if direction == 'right' and gate_idx + 1 < len(circ):
        pre_states[gate_idx+1] = op_action(new_mat, indices, pre_states[gate_idx])
    if direction == 'left' and gate_idx - 1 >= 0:
        post_states[gate_idx-1] = op_action(new_mat.conj().T, indices, post_states[gate_idx])

    return 1 - np.abs(np.trace(new_mat @ env)) 

def environment_state_prep(target_sv, circ=None, skeleton=None, context=None, quiet=True):
    """ uses environment tensors to optimize a circuit to prepare target_sv 
    either circ, skeleton or context.resume_state must be provided """
    context = RunContext() if context is None else context
    
    if context.resume: 
        circ = context.opt_state['circ']
        cost_list = context.opt_state['cost_list']
        pre_states = context.opt_state['pre_states']
        post_states = context.opt_state['post_states']
    else:
        if circ is None:
            circ = two_local_circ(skeleton)
        
        sim = StatevectorSim(circ)
        ansatz = sim.run(progress_bar=False).reshape(2**circ.num_sites)
        cost_list = [1-np.abs(target_sv.conj().T @ ansatz)]
        pre_states, post_states = cache_states(circ, target_sv)

    for step in range(context.step, context.max_iter):
        right_sweep = tqdm(range(len(circ)-1), disable=quiet)
        left_sweep = tqdm(reversed(range(1,len(circ))), disable=quiet)
        cost_list += [environment_update(circ, idx, pre_states, post_states, direction='right') for idx in right_sweep]
        cost_list += [environment_update(circ, idx, pre_states, post_states, direction='left') for idx in left_sweep]
        
        opt_state = {'step': step, 'circ': circ, 'cost_list': cost_list, 'pre_states': pre_states, 'post_states': post_states}
        if context.step_update(opt_state):
            break

    return circ, cost_list