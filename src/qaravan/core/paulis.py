import numpy as np
import itertools

pauli_I = np.array([[1,0],[0,1]], dtype=complex)
pauli_X = np.array([[0,1],[1,0]], dtype=complex)
pauli_Y = np.array([[0,-1.j],[1.j,0]], dtype=complex)
pauli_Z = np.array([[1,0],[0,-1]], dtype=complex)
pauli_mapping = {'i': pauli_I, 'x': pauli_X, 'y': pauli_Y, 'z': pauli_Z}

def pauli_commute(pauli1, pauli2):
    anticommute_count = 0
    for p1, p2 in zip(pauli1, pauli2):
        if p1 == 'i' or p2 == 'i':
            continue  
        if p1 != p2:
            anticommute_count += 1
    return anticommute_count % 2 == 0

def pauli_multiply(pauli1, pauli2):
    pauli_dict = {'i': 0, 'x': 1, 'y': 2, 'z': 3}
    inverse_pauli_dict = {0: 'i', 1: 'x', 2: 'y', 3: 'z'}

    epsilon = np.zeros((4, 4, 4), dtype=int)
    epsilon[1, 2, 3] = epsilon[2, 3, 1] = epsilon[3, 1, 2] = 1
    epsilon[3, 2, 1] = epsilon[1, 3, 2] = epsilon[2, 1, 3] = -1

    result_pauli = []
    phase = 1

    for p1, p2 in zip(pauli1, pauli2):
        i, j = pauli_dict[p1], pauli_dict[p2]

        if i == 0: 
            result_pauli.append(p2)
        elif j == 0:
            result_pauli.append(p1)
        elif i == j:  
            result_pauli.append('i')
        else:
            k = 6 - (i + j)  # Ensures X(1), Y(2), Z(3) → 1+2+3 = 6, so k = the missing index
            sign = epsilon[i, j, k]  # Get the correct sign
            result_pauli.append(inverse_pauli_dict[k])

            phase *= 1j * sign

    return "".join(result_pauli), phase

def pauli_strings(n):
    pauli_ops = 'ixyz'
    return [''.join(p) for p in itertools.product(pauli_ops, repeat=n)]

def random_pauli_string(n):
    chars = 'ixyz'
    out = ''
    for i in range(n):
        out += chars[np.random.choice(len(chars))]
    return out