from .tn import *
from numpy.random import uniform, normal

######################################
################ TI ##################
######################################

def one_copy_projector_ti_subspace(local_dimension, number_sites):
    """ returns r**number_sites where r is the effective local dimension """ 
    symmetry = np.concatenate([np.arange(number_sites, dtype=int), np.arange(number_sites, dtype=int)])
    permutation_list = [symmetry[i : i + number_sites] for i in range(number_sites)]
    length_permutation_list = len(permutation_list)
    cycle_counts = np.array([Permutation(permutation_list[i]).cycles for i in range(length_permutation_list)], dtype=int)
    return_value = np.sum(float(local_dimension) ** cycle_counts) / len(permutation_list)
    return return_value

class TI_MPS(MPS):
    def __init__(self, site, num_sites):
        super().__init__([site] * num_sites)
    
    def overlap(self, other, scaled=False): 
        """ if scaled, multiplies every transfer matrix by sqrt(r) where d is the effective local dimension """
        if scaled:
            scale = np.sqrt(one_copy_projector_ti_subspace(self.local_dim, self.num_sites)**(1/self.num_sites)) 
        else: 
            scale = 1.0
            
        tensor = ncon((self.sites[0], other.sites[0].conj()), ([-1,-3,1],[-2,-4,1])) * scale
        for i in range(1,self.num_sites):
            tensor = ncon((tensor, self.sites[i]), ([-1,-2,1,-3],[1,-4,-5]))
            tensor = ncon((tensor, other.sites[i].conj()), ([-1,-2,1,-3,2],[1,-4,2])) * scale
        return np.trace(tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2] * tensor.shape[3]))

######################################
############### HAAR #################
######################################
    
def random_unitary(n, local_dim=2): 
    # generates a random local_dim^n x local_dim^n unitary matrix
    a = np.random.rand(local_dim**n,local_dim**n) + 1j*np.random.rand(local_dim**n,local_dim**n)
    q,r = np.linalg.qr(a)
    d = np.diagonal(r)
    ph = d/np.abs(d)
    q *= ph
    return q

def haar_purity(num_sites, k_copies, local_dim=2, ti=False, scaled=False): 
    """ if scaled, returns purity * r**(num_sites*k_copies) where r is the effective local dimension """
    q = one_copy_projector_ti_subspace(local_dim, num_sites) if ti else local_dim**num_sites
    q_norm = q if scaled else 1
    factors = [(q+i)/q_norm for i in range(k_copies)]
    return np.math.factorial(k_copies)/np.prod(factors)

def haar_random_isometry(l_chi, r_chi=None, local_dim=2):
    r_chi = l_chi if r_chi is None else r_chi
    size = local_dim * l_chi
    temp = np.random.rand(size, size) + 1j * np.random.rand(size, size)
    temp = temp + temp.conj().T 
    return np.linalg.eigh(temp)[1][:, :r_chi]

#######################################
############## RMPS ###################
#######################################

def open_rmps_staggered(num_sites, chi, local_dim=2, distrib='isometric', param=1.0):
    sites = []
    for i in range(num_sites): 
        l_chi = min(local_dim**i, chi, local_dim**(num_sites-i))
        r_chi = min(local_dim**(i+1), chi, local_dim**(num_sites-1-i))
        shape = (l_chi, r_chi, local_dim)
        
        if distrib == 'isometric':
            site = haar_random_isometry(*shape).reshape(l_chi, local_dim, r_chi).transpose(0,2,1)
        elif distrib == 'uniform': 
            site = uniform(-1, 1, shape) + 1.j*uniform(-1, 1, shape)
        elif distrib == 'gaussian':
            site = normal(0, param, shape) + 1.j*normal(0, param, shape)
        else: 
            raise ValueError(f"{distrib} is not a valid distribution for site tensors")
        sites.append(site)
        
    mps = MPS(sites)
    if distrib != 'isometric': 
        mps.normalize()
    return mps 

def open_rmps_even(num_sites, chi, local_dim=2, distrib='isometric', param=1.0):
    sites = []
    for _ in range(num_sites):
        if distrib == 'isometric':
            site = haar_random_isometry(chi, chi, local_dim).reshape(chi, local_dim, chi).transpose(0,2,1) 
        elif distrib == 'uniform':
            site = uniform(-1, 1, (chi, chi, local_dim)) + 1.j * uniform(-1, 1, (chi, chi, local_dim))
        elif distrib == 'gaussian':
            site = normal(0, param, (chi, chi, local_dim)) + 1.j * normal(0, param, (chi, chi, local_dim))
        else: 
            raise ValueError(f"{distrib} is not a valid distribution for site tensors")
        sites.append(site)

    edge = np.zeros(chi)
    edge[0] = 1.0
    sites[0] = ncon((edge, sites[0]), ([1], [1,-1,-2]))[np.newaxis, :, :]
    sites[-1] = ncon((sites[-1], edge), ([-1,1,-2], [1]))[:, np.newaxis, :]
    mps = MPS(sites)
    mps.normalize()
    return mps
    
def periodic_rmps(num_sites, chi, local_dim=2, distrib="isometric", param=1.0): 
    sites = [haar_random_isometry(chi, chi, local_dim).reshape(chi, local_dim, chi).transpose(0,2,1) 
             for _ in range(num_sites)]
    mps = MPS(sites)
    mps.normalize()
    return mps

def ti_rmps(num_sites, chi, local_dim=2, distrib="isometric", param=1.0):
    site = haar_random_isometry(chi, chi, local_dim).reshape(chi, local_dim, chi).transpose(0,2,1)
    mps = TI_MPS(site, num_sites)
    mps.normalize()
    return mps

###################################################
############### MPOs ##############################
###################################################

def rand_mpo(n, chi, boundary='open', local_dim=2):
    if boundary == 'open': 
        sites = [np.random.rand(1,chi,local_dim,local_dim)] + [np.random.rand(chi,chi,local_dim,local_dim)
                     for _ in range(n-2)] + [np.random.rand(chi,1,local_dim,local_dim)]
    elif boundary == 'periodic': 
        sites = [np.random.rand(chi,chi,local_dim,local_dim) for _ in range(n)]
    else: 
        raise ValueError(f"Invalid boundary: {boundary}")
        
    return MPO(sites)

def all_zero_mps(n, local_dim=2):
    vec = np.zeros(local_dim)
    vec[0] = 1.0
    return MPS([vec[np.newaxis,np.newaxis,:]] * n)

def string_to_mps(string, local_dim): 
    sites = []
    for s in string: 
        vec = np.zeros(local_dim)
        vec[int(s)] = 1.0
        sites.append(vec[np.newaxis, np.newaxis, :])
    return MPS(sites)

def all_zero_mpdo(n, local_dim=2):
    mat = np.zeros((local_dim, local_dim))
    mat[0,0] = 1.0
    return MPDO([mat[np.newaxis,np.newaxis,:,:] for i in range(n)])