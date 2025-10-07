from itertools import product

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
    
def brickwall_skeleton(n, num_layers): 
    skeleton = []
    for _ in range(num_layers):
        for i in range(n//2): 
            skeleton.append((2*i,2*i+1))
        for i in range((n+1)//2 -1): 
            skeleton.append((2*i+1,2*i+2))
    return skeleton 

def brickwall_skeleton_2D(Lx, Ly):
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
    return skeleton