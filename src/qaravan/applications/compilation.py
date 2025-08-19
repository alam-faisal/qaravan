from qaravan.core.utils import *
from qaravan.core.gates import CNOT, CNOT3, embed_operator, Gate
from qaravan.core.param_gates import U, U01, RZ
from qaravan.core.paulis import pauli_Z
from qaravan.core.circuits import Circuit, circ_to_mat
from qaravan.tensorQ.statevector_sim import StatevectorSim

from scipy.linalg import expm
from itertools import product, combinations
from scipy.optimize import minimize
from tqdm import tqdm
from numpy.linalg import qr 
import copy
import torch

# only for qubits

################################################
################ CIRCUIT ANSATZES ##############
################################################

def dressed_cnot_circ(skeleton, num_qubits, params, local_dim=2):
    single_site = U if local_dim == 2 else U01
    two_site = CNOT if local_dim == 2 else CNOT3 
    
    counter = 0
    gate_list = []
    for pair in skeleton: 
        gate_list.append(single_site(pair[0], *params[3*counter:3*(counter+1)]))
        counter += 1

        gate_list.append(single_site(pair[1], *params[3*counter:3*(counter+1)]))
        counter += 1

        gate_list.append(two_site(pair, num_qubits))

    for i in range(num_qubits):
        gate_list.append(single_site(i, *params[3*counter:3*(counter+1)]))
        counter += 1

    return Circuit(gate_list, num_qubits)

def dressed_cnot_skeletons(num_sites, num_cnots, orientation=False, a2a=False):
    if not a2a: 
        pairs = [(i, i+1) for i in range(num_sites-1)]
    else: 
        pairs = [(i,j) for i in range(num_sites) for j in range(num_sites) if i != j]

    if orientation: 
        pairs += [pair[::-1] for pair in pairs]

    return list(product(pairs, repeat=num_cnots)) 
    
def brickwall_skeleton(n, num_layers): 
    skeleton = []
    for _ in range(num_layers):
        for i in range(n//2): 
            skeleton.append((2*i,2*i+1))
        for i in range((n+1)//2 -1): 
            skeleton.append((2*i+1,2*i+2))
    return skeleton 

def k_body_z(n, k): 
    ops = []
    for indices in combinations(range(n), k): 
        name = "".join("Z" if i in indices else "I" for i in range(n))
        mat = embed_operator(n, list(indices), [pauli_Z] * k, dense=True)
        ops.append((name, mat))
    return ops

def rz_ansatz(n, params, body=1): 
    gate_list = [RZ(i, params[i]) for i in range(n)]
    for k in range(2,body+1):
        generators = k_body_z(n, k)
        gates = []
        for i, (name, mat) in enumerate(generators):
            u = expm(-1.j * params[len(gate_list)+i] * mat)
            gates.append(Gate('R'+name, np.arange(n), u))
        gate_list += gates

    return Circuit(gate_list, n)

def random_circ(n, depth): 
    skeleton = brickwall_skeleton(n,depth)
    num_cnots = len(skeleton)
    num_params = (num_cnots * 6) + (n * 3)
    return dressed_cnot_circ(skeleton, n, np.random.rand(num_params))

###############################################
############# COMPILERS #######################
###############################################

def msp_infidelity(params, skeleton, in_states, out_states, backend="numpy"):
    """ assuming qubits here """
    num_qubits = int(np.log2(len(in_states[0])))
    ansatz = dressed_cnot_circ(skeleton, num_qubits, params, local_dim=2)
    
    infid = 0
    for in_state, out_state in zip(in_states, out_states):
        sim = StatevectorSim(ansatz, backend=backend, init_state=in_state)
        sim.run(progress_bar=False)
        ansatz_sv = sim.get_statevector()

        if backend == "torch":
            target_sv = torch.tensor(out_state, dtype=torch.complex128)
            infid += 1 - torch.abs(torch.dot(ansatz_sv.conj(), target_sv))**2
        else:
            infid += 1 - np.abs(np.dot(ansatz_sv.conj(), out_state))**2 
                  
    return infid / len(in_states)

def msp_via_autodiff(in_states, out_states, cnot_skeleton, init_params=None, lr=0.05, 
                     num_sweeps=100, threshold=2e-8):
    
    if init_params is None:
        num_qubits = int(np.log2(len(in_states[0])))
        num_cnots = len(cnot_skeleton)
        num_params = (num_cnots * 6) + (num_qubits * 3)
        init_params = torch.nn.Parameter(torch.randn(num_params))
    
    elif not isinstance(init_params, torch.nn.Parameter):
        init_params = torch.nn.Parameter(torch.tensor(init_params, dtype=torch.float64))

    optimizer = torch.optim.Adam([init_params], lr=lr)
    context = RunContext(
        progress_interval=10,
        max_iter=num_sweeps,
        checkpoint_file="checkpoint.pickle",
        checkpoint_interval=100,
        resume=False,
        convergence_check=True,
        stop_ratio=1e-8, 
        stop_absolute=threshold,
    )

    cost_list = []
    run_state = {'cost_list': cost_list, 'step': context.step}
    
    while True: 
        cost = msp_infidelity(init_params, cnot_skeleton, in_states, out_states, backend="torch")
        cost_list.append(cost.item())
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        run_state["step"] += 1
        run_state["cost_list"] = cost_list

        if context.step_update(run_state):
            print(f"Terminating at step {run_state['step']} with cost {cost_list[-1]}")
            break

    return init_params, run_state

def msp_environment(in_state, ansatz, out_state, gate_idx): 
    d = ansatz.local_dim
    env = np.zeros((d,d), dtype=complex)
    for i in range(d): 
        for j in range(d): 
            a_copy = copy.deepcopy(ansatz)
            mat = np.zeros((d,d))
            mat[i,j] = 1.0
            a_copy.update_gate(gate_idx, mat)
            env[i,j] = out_state.conj().T @ a_copy.to_matrix() @ in_state 
            
    return env.T 

def msp_via_environments(in_states, ansatz, out_states, num_sweeps=100, quiet=True, threshold=2e-8): 
    term_cond = None
    cost_list = []
    for sweep in tqdm(range(num_sweeps)):
        for gate_idx in range(len(ansatz.gate_list)): 
            if ansatz.gate_list[gate_idx].name[0] == 'U': 
                gate = ansatz.gate_list[gate_idx]
                
                env_list = []
                for in_state,out_state in zip(in_states, out_states):
                    env = msp_environment(in_state, ansatz, out_state, gate_idx) 
                    env_list.append(env)
                    
                env = sum(env_list)

                x,s,yh = np.linalg.svd(env) 
                new_gate = yh.conj().T @ x.conj().T 
                ansatz.update_gate(gate_idx, new_gate)

                new_cost = len(in_states)-np.trace(env@new_gate).real
                if new_cost < 0: 
                    if np.abs(new_cost) < 1e-6: 
                        new_cost = 1e-16 
                    else: 
                        print("Cost has become negative somehow")
        
        cost_list.append(np.sqrt(new_cost))
        if sweep > 1 and (cost_list[-2] - cost_list[-1]) / cost_list[-2] < 1e-4:
            term_cond = "cost function has plateaued"
            break
        if cost_list[-1] < threshold:
            term_cond = "cost function has converged"
            break
        if not quiet: 
            if sweep % 10 == 0: 
                print(cost_list[-1])
    
    term_cond = "reached maximum number of sweeps" if term_cond is None else term_cond
    print(term_cond)
    return ansatz, cost_list, term_cond

def msp_job(in_states, out_states, skeleton, local_dim=2, max_sweeps=500, max_instances=50, threshold=2e-8): 
    num_cnots = len(skeleton)
    num_sites = int(np.ceil(np.log(len(in_states[0]))/np.log(local_dim)))
    num_params = (num_cnots * 6) + (num_sites * 3)
    
    sols = []
    data = []
    for i in range(max_instances):
        params = np.random.rand(num_params)
        ansatz = dressed_cnot_circ(skeleton, num_sites, params, local_dim) 
        new_ansatz, cost_list, term_cond = msp_via_environments(in_states, ansatz, out_states, 
                                            num_sweeps=max_sweeps, threshold=threshold)
        sols.append(new_ansatz)
        data.append({'sol':new_ansatz, 'cost_list':cost_list, 'term_cond': term_cond})

        print(f"After optimization: {cost_list[-1]}")
        if cost_list[-1] < threshold: 
            break
            
    return sols, data

def construct_unitary(in_strings, out_states, local_dim=2):
    in_mat = np.array([string_to_sv(in_str, local_dim) for in_str in in_strings])
    out_mat = np.array(out_states)
    A = out_mat.T @ in_mat
    Q, _ = qr(A, mode='reduced')
    
    true_cols = [int(in_str, local_dim) for in_str in in_strings]
    for col in true_cols: 
        Q[:,col] = A[:,col]
    return Q

def unitary_embedding(in_strings, out_states, local_dim=2): 
    # deprecated
    print("This function is deprecated. Use construct_unitary instead.")

    n = len(out_states[0])
    
    in_states = [string_to_sv(in_str, local_dim) for in_str in in_strings]
    #A = (np.array(out_states).conj().T @ np.array(in_states))
    A = (np.array(out_states).T @ np.array(in_states))
    
    true_rows = [int(in_str, local_dim) for in_str in in_strings]
    space = scipy.linalg.null_space(A.T)
    
    A_row = 0
    space_row = 0
    while space_row < n-len(in_strings): 
        if A_row not in true_rows: 
            A[:,A_row] = space[:,space_row]
            space_row += 1
        A_row += 1
        
    return A 

############################################## 
###### DEPRECATED / BAD FUNCTIONS ############
##############################################

def msp_cost(params, skeleton, vecs, target): 
    num_qubits = int(np.log2(len(vecs[0])))
    ansatz_mat = circ_to_mat(dressed_cnot_circ(skeleton, num_qubits, params))
    
    c = 0.0 
    if type(target) == np.ndarray:
        for vec in vecs: 
            c += np.linalg.norm(ansatz_mat @ vec - target @ vec)
    else:
        for vec,out in zip(vecs, target): 
            c += np.linalg.norm(ansatz_mat @ vec - out)
        
    return c / (len(vecs)) 

def msp_optimize(vecs, target, skeleton, num_reps=10, quiet=True): 
    """ 
    target can either be the matrix or the list of output statevectors 
    vecs are the list of input statevectors
    """
    print("SciPy optimizers are horrible. Use msp_via_autodiff or msp_via_environments instead.")

    num_cnots = len(skeleton)
    num_qubits = int(np.log2(len(vecs[0])))
    num_params = (num_cnots * 6) + (num_qubits * 3)
    
    results = []
    vals = []
    for i in tqdm(range(num_reps)): 
        params = np.random.rand(num_params)
        result = minimize(msp_cost, x0=params, args=(skeleton, vecs, target), 
                                 method='L-BFGS-B', options={'maxfun':5000, 'maxiter':5000, 'disp':False})
        results.append(result)
        vals.append(result.fun)
        if not quiet:
            print(result.fun)

    print(f"Best cost with {num_cnots} CNOTs is: {min(vals)}")
    return results, vals

