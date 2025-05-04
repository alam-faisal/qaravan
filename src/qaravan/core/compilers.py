from .utils import *
from .gates import CNOT, CNOT3, embed_operator, Gate
from .param_gates import U, U01, RZ
from .paulis import pauli_Z
from .circuits import Circuit
from scipy.linalg import expm
from itertools import product, combinations
from scipy.optimize import minimize
from tqdm import tqdm
from numpy.linalg import qr 

# only for qubits

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

def dressed_cnot_skeletons(num_sites, num_cnots, orientation=False):
    nn_pairs = [(i+1, i) for i in range(num_sites-1)]
    if orientation: 
        nn_pairs += [(i, i+1) for i in range(num_sites-1)]
    skeletons = list(product(nn_pairs, repeat=num_cnots))
    return skeletons

def msp_cost(params, skeleton, vecs, target): 
    num_qubits = int(np.log2(len(vecs[0])))
    ansatz_mat = circ_to_mat(dressed_cnot_circ(skeleton, num_qubits, params))
    
    c = 0.0 
    
    if type(target) == np.ndarray:
        for vec in vecs: 
            c += np.linalg.norm(ansatz_mat @ vec - mat @ vec)
    else:
        for vec,out in zip(vecs, target): 
            c += np.linalg.norm(ansatz_mat @ vec - out)
        
    return c / (len(vecs)) 

def msp_optimize(vecs, target, skeleton, num_reps=10, quiet=True): 
    num_cnots = len(skeleton)
    num_qubits = int(np.log2(len(vecs[0])))
    num_params = (num_cnots * 6) + (num_qubits * 3)
    
    results = []
    vals = []
    for i in range(num_reps): 
        params = np.random.rand(num_params)
        result = minimize(msp_cost, x0=params, args=(skeleton, vecs, target), 
                                 method='BFGS')
        results.append(result)
        vals.append(result.fun)
        if not quiet:
            print(result.fun)

    print(f"Best cost with {num_cnots} CNOTs is: {min(vals)}")
    return results, vals

def brickwall_skeleton(n, num_layers): 
    skeleton = []
    for _ in range(num_layers):
        for i in range(n//2): 
            skeleton.append((2*i,2*i+1))
        for i in range((n+1)//2 -1): 
            skeleton.append((2*i+1,2*i+2))
    return skeleton 

def environment(in_state, ansatz, out_state, gate_idx): 
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

def ssp_opt(in_state, ansatz, out_state, num_sweeps=100): 
    cost_list = []
    term_cond = None
    for sweep in range(num_sweeps):
        for gate_idx in range(len(ansatz.gate_list)): 
            if ansatz.gate_list[gate_idx].name[0] == 'U': 
                gate = ansatz.gate_list[gate_idx]
                env = environment(in_state, ansatz, out_state, gate_idx) 

                x,s,yh = np.linalg.svd(env) 
                new_gate = yh.conj().T @ x.conj().T 
                ansatz.update_gate(gate_idx, new_gate)

                new_cost = 1-np.trace(env@new_gate).real
                if new_cost < 0: 
                    if np.abs(new_cost) < 1e-6: 
                        new_cost = 1e-16 
                    else: 
                        print("Cost has become negative somehow")
                
        cost_list.append(np.sqrt(new_cost))
        if sweep > 1 and (cost_list[-2] - cost_list[-1]) / cost_list[-2] < 1e-3:
            term_cond = "cost function has plateaued"
            break
        if cost_list[-1] < 1e-6:
            term_cond = "cost function has converged"
            break
            
        
    print(term_cond)
    return ansatz, cost_list, term_cond 

def msp_opt(in_states, ansatz, out_states, num_sweeps=100, quiet=True, threshold=2e-8): 
    term_cond = None
    cost_list = []
    for sweep in tqdm(range(num_sweeps)):
        for gate_idx in range(len(ansatz.gate_list)): 
            if ansatz.gate_list[gate_idx].name[0] == 'U': 
                gate = ansatz.gate_list[gate_idx]
                
                env_list = []
                for in_state,out_state in zip(in_states, out_states):
                    env = environment(in_state, ansatz, out_state, gate_idx) 
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
        new_ansatz, cost_list, term_cond = msp_opt(in_states, ansatz, out_states, num_sweeps=max_sweeps, threshold=threshold)
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
