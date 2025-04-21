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

def environment_update(circ, gate_idx, pre_state, post_state, target_sv, init_sv, direction='decreasing'): 
    indices = circ[gate_idx].indices
    env = partial_overlap(pre_state, post_state, skip=indices)
    x,s,yh = np.linalg.svd(env)
    new_mat = yh.conj().T @ x.conj().T
    circ.update_gate(gate_idx, new_mat)

    if direction == 'decreasing': 
        if gate_idx == 1: 
            pre_state = init_sv 
        else:
            pre_state = op_action(circ[gate_idx-1].matrix.conj().T, circ[gate_idx-1].indices, pre_state)   # undo the next gate        
        post_state = op_action(new_mat.conj().T, indices, post_state)

    else: 
        pre_state = op_action(new_mat, indices, pre_state)        
        if gate_idx == len(circ)-2: 
            post_state = target_sv
        else:
            post_state = op_action(circ[gate_idx+1].matrix, circ[gate_idx+1].indices, post_state)   # undo the next gate

    return 1 - np.abs(np.trace(new_mat @ env)), pre_state, post_state

def environment_state_prep(target_sv, init_sv=None, circ=None, skeleton=None, context=None, quiet=True):
    """ uses environment tensors to optimize a circuit to prepare target_sv 
    either circ, skeleton or context.resume_state must be provided """
    context = RunContext() if context is None else context
    
    if context.resume: 
        circ = context.run_state['circ']
        cost_list = context.run_state['cost_list']
        pre_state = context.run_state['pre_state']
        post_state = context.run_state['post_state']
    else:
        if circ is None:
            circ = two_local_circ(skeleton)
        
        sim = StatevectorSim(circ, init_state=init_sv)
        ansatz = sim.run(progress_bar=False).reshape(2**circ.num_sites)
        cost_list = [1-np.abs(target_sv.conj().T @ ansatz)]

        pre_state = op_action(circ[-1].matrix.conj().T, circ[-1].indices, ansatz)   # undo the last gate 
        post_state = target_sv

    init_sv = all_zero_sv(circ.num_sites, dense=True) if init_sv is None else init_sv
    for step in range(context.step, context.max_iter):
        left_sweep = tqdm(reversed(range(1,len(circ))), disable=quiet, desc='Left sweep')
        right_sweep = tqdm(range(len(circ)-1), disable=quiet, desc='Right sweep')

        for i in left_sweep: 
            cost, pre_state, post_state = environment_update(circ, i, pre_state, post_state, target_sv, init_sv, direction='decreasing')
            cost_list.append(cost)

        for i in right_sweep:
            cost, pre_state, post_state = environment_update(circ, i, pre_state, post_state, target_sv, init_sv, direction='increasing')
            cost_list.append(cost)
        
        run_state = {'step': step, 'circ': circ, 'cost_list': cost_list, 'pre_state': pre_state, 'post_state': post_state}
        if context.step_update(run_state):
            break

    return circ, cost_list