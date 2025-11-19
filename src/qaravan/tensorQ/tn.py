from ncon_torch import ncon
import numpy as np
import copy
from scipy.linalg import svd, block_diag

###################################
##### Tensor methods ##############
###################################

default_msvr = 1e-8
def site_svd(site, which='left', msvr=None, max_dim=None, return_sv=False):
    """
    does an asymmetric SVD, keeping spin indice(s) on 'which' side, truncates depending on either max_dim 
    or min_sv_ratio
    """
    sh = site.shape
    is_mpo = len(sh) == 4
    
    if which == 'left': 
        if is_mpo:
            data = site.transpose(0,2,3,1).reshape(sh[0] * sh[2] * sh[3], sh[1])
        else: 
            data = site.transpose(0,2,1).reshape(sh[0] * sh[2], sh[1])
            
    else: 
        if is_mpo: 
            data = site.reshape(sh[0], sh[1] * sh[2] * sh[3])
        else: 
            data = site.reshape(sh[0], sh[1] * sh[2]) 
    
    u,s,vh = svd(data, full_matrices=False, lapack_driver='gesvd') 
    if msvr is not None: 
        s = s[s>msvr*s[0]]
    elif max_dim is not None:
        dim = min(max_dim, len(s[s>default_msvr*s[0]]))
        s = s[:dim]
        
    u = u[:,:len(s)] 
    vh = vh[:len(s),:]
    
    if which == 'left': 
        if is_mpo:
            left_tensor = u.reshape(sh[0], sh[2], sh[3], len(s)).transpose(0,3,1,2)
        else: 
            left_tensor = u.reshape(sh[0], sh[2], len(s)).transpose(0,2,1)
        right_tensor = np.diag(s) @ vh if not return_sv else vh
            
    else: 
        left_tensor = u @ np.diag(s) if not return_sv else u
        if is_mpo:
            right_tensor = vh.reshape(len(s), sh[1], sh[2], sh[3])
        else: 
            right_tensor = vh.reshape(len(s), sh[1], sh[2])
    
    if return_sv: 
        return left_tensor, right_tensor, s
    else:
        return left_tensor, right_tensor
    
def decompose_site(data): 
    """ 
    decomposes sites of an MPS or MPDO after action of a two-qubit gate 
    data.shape = (first_top_spin, first_bottom_spin, second_top_spin, second_bottom_spin, 
                    left_bond, right_bond)
    """
    sh = data.shape
    is_mpo = len(sh) == 6
    
    if is_mpo:
        data = data.transpose(0,1,4,2,3,5).reshape(sh[0]*sh[1]*sh[4], sh[2]*sh[3]*sh[5])
    else:
        data = data.transpose(0,2,1,3).reshape(sh[0]*sh[2], sh[1]*sh[3])
    
    u,s,vh = svd(data, full_matrices=False, lapack_driver='gesvd')    
    u, vh = u @ np.diag(np.sqrt(s)), np.diag(np.sqrt(s)) @ vh

    if is_mpo:
        top_site = u.reshape(sh[0],sh[1],sh[4],len(s)).transpose(2,3,0,1)
        bottom_site = vh.reshape(len(s),sh[2],sh[3],sh[5]).transpose(0,3,1,2)
    else:
        top_site = u.reshape(sh[0],sh[2],len(s)).transpose(1,2,0)
        bottom_site = vh.reshape(len(s),sh[1],sh[3]).transpose(0,2,1)
        
    return top_site, bottom_site

def decompose_three_site(data): 
    """ 
    decomposes sites of an MPDO after action of a three-site gate 
    data.shape = (first_top_spin, first_bottom_spin, second_top_spin, second_bottom_spin, 
                    third_top_spin, third_bottom_spin, left_bond, right_bond)
    """
    sh = data.shape
    data = data.transpose(0,1,2,4,3,5,6,7).reshape(sh[0],sh[1],sh[2]*sh[4],sh[3]*sh[5], sh[6], sh[7])
    
    top_site, bottom_site = decompose_site(data)
    bottom_site = bottom_site.reshape(bottom_site.shape[0], 
                                      sh[7], sh[2], sh[4], sh[3], sh[5]).transpose(2,4,3,5,0,1)
    middle_site, bottom_site = decompose_site(bottom_site)
    return top_site, middle_site, bottom_site 

def decompose_four_site(data): 
    """ 
    decomposes sites of an MPDO after action of a four-site gate 
    data.shape = (first_top_spin, first_bottom_spin, second_top_spin, second_bottom_spin, 
                    third_top_spin, third_bottom_spin, fourth_top_spin, fourth_bottom_spin, 
                    left_bond, right_bond)
    """
    sh = data.shape
    data = data.transpose(0,2,1,3,4,6,5,7,8,9).reshape(sh[0]*sh[2],sh[1]*sh[3],sh[4]*sh[6],sh[5]*sh[7], sh[8], sh[9])
    
    top_half, bottom_half = decompose_site(data)
    
    top_half = top_half.reshape(sh[8], top_half.shape[1], sh[0], sh[2], sh[1], sh[3]).transpose(2,4,3,5,0,1)
    bottom_half = bottom_half.reshape(bottom_half.shape[0], sh[9], sh[4], sh[6], sh[5], sh[7]).transpose(2,4,3,5,0,1)
    
    first_site, second_site = decompose_site(top_half)
    third_site, fourth_site = decompose_site(bottom_half)    
    return first_site, second_site, third_site, fourth_site 


############################################
############### CLASSES ####################
############################################

class TensorNetwork:
    def __init__(self, sites):
        self.sites = sites
        self.num_sites = len(sites)
        self.local_dim = sites[0].shape[-1]
        
    def __add__(self, other): 
        raise NotImplementedError("Subclasses must implement this method")
    
    def __mul__(self, scalar): 
        sites = copy.deepcopy(self.sites)
        sites[0] = sites[0] * scalar
        return type(self)(sites)
        
    def __matmul__(self, other):
        raise NotImplementedError("Subclasses must implement this method")
        
    def get_skeleton(self): 
        return [site.shape for site in self.sites]
    
    def get_max_dim(self):
        return max(max(shape) for shape in self.get_skeleton())
    
    def get_site_norms(self): 
        return [np.linalg.norm(np.ravel(site)) for site in self.sites]
    
    def conj(self): 
        sites = [copy.deepcopy(site).conj() for site in self.sites]
        return type(self)(sites)

    def __getitem__(self, index):
        return self.sites[index]
        
class MPS(TensorNetwork): 
    """ a site is a np.array of shape (left_bond_dim, right_bond_dim, local_dim) """
    def __init__(self, sites):
        super().__init__(sites)
        self.center = None 
        self.right_envs = None 
        self.left_envs = None 
    
    def __matmul__(self, other):
        if isinstance(other, MPS):
            return self.overlap(other)
        else:
            raise TypeError("Unsupported operand type for @")
        
    def compute_right_envs(self, other=None, scale=1):
        """ usually pre-computed and reused for efficiency """
        if self.right_envs is not None:
            return self.right_envs
    
        other = self if other is None else other

        right_envs = [None] * (self.num_sites)
        right_envs[-1] = np.array([[1.0]]) 
        for i in range(self.num_sites-1, 0, -1):
            right_envs[i-1] = contract_mps_env(right_envs[i], self.sites[i], other.sites[i], right=True) * scale

        self.right_envs = right_envs
        return right_envs
    
    def compute_left_envs(self, other=None, scale=1):
        """ usually pre-computed and reused for efficiency """
        if self.left_envs is not None:
            return self.left_envs

        other = self if other is None else other

        left_envs = [None] * (self.num_sites)
        left_envs[0] = np.array([[1.0]]) 
        
        for i in range(1, self.num_sites):
            left_envs[i] = contract_mps_env(left_envs[i-1], self.sites[i-1], other.sites[i-1], right=False) * scale
        
        self.left_envs = left_envs
        return left_envs
    
    def overlap(self, other, scaled=False):   
        scale = np.sqrt(self.local_dim) if scaled else 1.0
        self.compute_right_envs(other, scale=scale)
        tensor = ncon((self.sites[0], other.sites[0].conj()), ([1,-1,2],[1,-2,2])) * scale
        return ncon((tensor, self.right_envs[0]), ([1,2],[1,2]))
         
    def norm(self): 
        return self.overlap(self)
    
    def normalize(self): 
        n = self.norm()
        self.sites = [site/np.sqrt(np.abs(n))**(1/self.num_sites) for site in self.sites]
        # need to reset environments after normalization
        self.left_envs = None
        self.right_envs = None
    
    def canonize(self, center): 
        """ orthogonalize self.state at center; 
        upto center-1 we have left-canonical; 
        from center+1 we have right-canonical 
        if center == 0 and MPS is normalized, then all sites are right-canonical 
        if center == num_sites-1 and MPS is normalized, then all sites are left-canonical
        """
        sites = copy.deepcopy(self.sites)
        if sites[0].shape[0] > 1:
            raise NotImplementedError("canonization for periodic boundaries has not been implemented")

        left_start = 0 if self.center is None else self.center
        right_start = self.num_sites - 1 if self.center is None else self.center

        for i in range(right_start, center, -1):
            left_tensor, right_tensor = site_svd(sites[i], 'right', None, None, False)
            sites[i] = right_tensor
            sites[i-1] = ncon((sites[i-1], left_tensor), ([-1,1,-3], [1,-2]))

        for i in range(left_start, center):
            left_tensor, right_tensor = site_svd(sites[i], 'left', None, None, False)
            sites[i] = left_tensor
            sites[i+1] = ncon((right_tensor, sites[i+1]), ([-1,1], [1,-2,-3]))

        self.center = center
        self.sites = sites
        self.left_envs = None
        self.right_envs = None
        
    def compress(self, msvr=None, max_dim=None):
        sites = copy.deepcopy(self.sites)
        if sites[0].shape[0] > 1: 
            raise NotImplementedError("compression for periodic boundaries has not been implemented")

        # first we orthogonalize
        for i in range(self.num_sites-1, 0, -1):
            left_tensor, right_tensor = site_svd(sites[i], 'right', None, None)
            sites[i] = right_tensor
            sites[i-1] = ncon((sites[i-1], left_tensor), ([-1,1,-3], [1,-2]))
        
        # then we sweep down and truncate
        for i in range(0, self.num_sites-1):
            left_tensor, right_tensor = site_svd(sites[i], 'left', msvr, max_dim)
            sites[i] = left_tensor
            sites[i+1] = ncon((right_tensor, sites[i+1]), ([-1,1], [1,-2,-3]))
        
        # then we sweep up and truncate
        for i in range(self.num_sites-1, 0, -1):
            left_tensor, right_tensor = site_svd(sites[i], 'right', msvr, max_dim)
            sites[i] = right_tensor
            sites[i-1] = ncon((sites[i-1], left_tensor), ([-1,1,-3], [1,-2]))

        return MPS(sites)
    
    def fast_measure(self, meas_sites):
        """ 
        currently supports qubits in Z basis, assumes state is normalized
        built for efficient sampling of non-contiguous sites 
        """
        self.compute_right_envs()

        cur_idx = 0
        cur_left = np.array([[1.0]]) 
        outcome = ''
        for meas_site in meas_sites:
            for i in range(cur_idx, meas_site):
                cur_left = contract_mps_env(cur_left, self.sites[i], right=False)

            left_tensor = contract_mps_env(cur_left, self.sites[meas_site], op=np.array([[1,0],[0,0]]), right=False)
            cur_right_env = self.right_envs[meas_site]
            prob_0 = ncon((left_tensor, cur_right_env), ([1,2],[1,2])).real

            rand = np.random.rand()
            if rand < prob_0:
                outcome += '0'
                proj_op = np.array([[1,0],[0,0]]) / prob_0
            else:
                outcome += '1'
                proj_op = np.array([[0,0],[0,1]]) / (1 - prob_0)

            cur_left = contract_mps_env(cur_left, self.sites[meas_site], op=proj_op, right=False)
            cur_idx = meas_site + 1
        
        return outcome
    
    def one_rdm(self, site_idx):
        self.compute_left_envs()
        self.compute_right_envs()

        le = self.left_envs[site_idx]
        re = self.right_envs[site_idx]
        site = self.sites[site_idx]

        le_con = ncon((le, site.conj()), ([-1,1],[1,-2,-3]))
        re_con = ncon((re, site), ([1,-2],[-1,1,-3]))

        return ncon((le_con, re_con), ([1,2,-2],[1,2,-1]))
    
    def one_local_expectation(self, site_idx, op):
        return np.trace(op @ self.one_rdm(site_idx)).real

    def to_vector(self):
        return contract_sites(self.sites)[0,0,:].reshape(self.local_dim**self.num_sites)
    
    def evaluate(self, basis): 
        return 0.0       
    
    def __str__(self):
        desc = "MPS with {} sites, local dim {}, max bond dim {}\n".format(
            self.num_sites, self.local_dim, self.get_max_dim())
        for i, site in enumerate(self.sites):
            desc += " Site {}: shape {}\n".format(i, site.shape)
        return desc
        
class MPO(TensorNetwork): 
    """ a site is a np.array of shape (left_bond_dim, right_bond_dim, top_local_dim, bottom_local_dim) """
    def __init__(self, sites):
        super().__init__(sites)
        
    def __add__(self, other): 
        new_sites = []
        
        for i, (s,o) in enumerate(zip(self.sites, other.sites)):
            if i == 0: 
                site = np.zeros((1, s.shape[1]+o.shape[1], self.local_dim, self.local_dim), dtype=complex)
                for i in range(self.local_dim):
                    for j in range(self.local_dim): 
                        site[0,:,i,j] = np.hstack((s[0,:,i,j], o[0,:,i,j]))
                        
            elif i == self.num_sites - 1: 
                site = np.zeros((s.shape[0]+o.shape[0], 1, self.local_dim, self.local_dim), dtype=complex)
                for i in range(self.local_dim):
                    for j in range(self.local_dim): 
                        site[:,0,i,j] = np.hstack((s[:,0,i,j], o[:,0,i,j]))
            
            else: 
                site = np.zeros((s.shape[0]+o.shape[0], s.shape[1]+o.shape[1], self.local_dim, self.local_dim), dtype=complex)
                for i in range(self.local_dim):
                    for j in range(self.local_dim): 
                        site[:,:,i,j] = block_diag(s[:,:,i,j], o[:,:,i,j])

            new_sites.append(site)
        return type(self)(new_sites)
    
    def __matmul__(self, other):
        if isinstance(other, MPS):
            return self.act(other)
        elif isinstance(other, MPO):
            return self.compose(other)
        else:
            raise TypeError("Unsupported operand type for @")
            
    def act(self, mps):
        new_sites = []
        for site_a, site_b in zip(self.sites, mps.sites):
            ldim = site_a.shape[0] * site_b.shape[0]
            rdim = site_a.shape[1] * site_b.shape[1]
            new_site = ncon((site_a, site_b), ([-1,-3,-5,1],[-2,-4,1])).reshape(ldim, rdim, mps.local_dim)
            new_sites.append(new_site)
        return MPS(new_sites)
    
    def compose(self, mpo):
        new_sites = []
        for site_a, site_b in zip(self.sites, mpo.sites):
            ldim = site_a.shape[0] * site_b.shape[0]
            rdim = site_a.shape[1] * site_b.shape[1]
            sdim = self.local_dim
            new_site = ncon((site_a, site_b), ([-1,-3,-5,1],[-2,-4,1,-6])).reshape(ldim, rdim, sdim, sdim)
            new_sites.append(new_site)
        return type(self)(new_sites)
    
    def overlap(self, other, scaled=True): 
        scale = self.local_dim if scaled else 1.0
        tensor = ncon((self.sites[0], other.sites[0]), ([-1,-3,1,2],[-2,-4,2,1]))/scale
        for i in range(1,self.num_sites):
            tensor = ncon((tensor, self.sites[i]), ([-1,-2,1,-3],[1,-4,-5,-6]))
            tensor = ncon((tensor, other.sites[i]), ([-1,-2,1,-3,2,3],[1,-4,3,2]))/scale
        return np.trace(tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2] * tensor.shape[3]))
    
    def trace(self, scaled=True): 
        scale = self.local_dim if scaled else 1.0
        if self.num_sites == 1:
            return np.trace(self.sites[0])
        else: 
            tr = ncon((self.sites[0]), ([-1,-2,1,1]))/scale
            for i in range(1,self.num_sites): 
                tr = ncon((tr, self.sites[i]), ([-1,1],[1,-2,2,2]))/scale
            return np.trace(tr)
    
    def to_matrix(self): 
        tensor = self.sites[0]
        for i in range(1,self.num_sites):
            left_dim = tensor.shape[0]
            right_dim = self.sites[i].shape[1]
            local_dim = self.local_dim**(i+1)
            tensor = ncon((tensor, self.sites[i]), ([-1,1,-3,-5],[1,-2,-4,-6])).reshape(left_dim,right_dim,local_dim,local_dim)
        return ncon((tensor), (1,1,-1,-2))
    
    def compress(self, msvr=None, max_dim=None):
        sites = copy.deepcopy(self.sites)
        if sites[0].shape[0] > 1: 
            raise NotImplementedError("compression for periodic boundaries has not been implemented")

        # first we orthogonalize
        for i in range(self.num_sites-1, 0, -1):
            left_tensor, right_tensor = site_svd(sites[i], 'right', None, None)
            sites[i] = right_tensor
            sites[i-1] = ncon((sites[i-1], left_tensor), ([-1,1,-3,-4], [1,-2]))
        
        # then we sweep down and truncate
        for i in range(0, self.num_sites-1):
            left_tensor, right_tensor = site_svd(sites[i], 'left', msvr, max_dim)
            sites[i] = left_tensor
            sites[i+1] = ncon((right_tensor, sites[i+1]), ([-1,1], [1,-2,-3,-4]))
        
        # then we sweep up and truncate
        for i in range(self.num_sites-1, 0, -1):
            left_tensor, right_tensor = site_svd(sites[i], 'right', msvr, max_dim)
            sites[i] = right_tensor
            sites[i-1] = ncon((sites[i-1], left_tensor), ([-1,1,-3,-4], [1,-2]))

        return type(self)(sites)
    
class MPDO(MPO):
    def __init__(self, sites):
        super().__init__(sites)
    
    def partial_trace(self, indices): 
        sites = copy.deepcopy(self.sites)
        offset = 0

        for i in indices:
            cur_idx = i-offset
            sites[cur_idx] = ncon((sites[cur_idx]), ([-1,-2,1,1]))

            if cur_idx ==  len(sites)-1: 
                sites[cur_idx-1] = ncon((sites[cur_idx-1], sites[cur_idx]), ([-1,1,-3,-4],[1,-2]))            
            else: 
                sites[cur_idx+1] = ncon((sites[cur_idx], sites[cur_idx+1]), ([-1,1],[1,-2,-3,-4])) 

            sites.pop(cur_idx)
            offset += 1

        return MPDO(sites)
    
    def apply_isometry(self, site_indices, isometries): 
        """ contracts with isometries """
        site_indices = [site_indices] if type(site_indices) == int else site_indices
        isometries = [isometries] * len(site_indices) if type(isometries) == np.ndarray else isometries
        sites = copy.deepcopy(self.sites)
        
        for site_idx, isometry in zip(site_indices, isometries): 
            sites[site_idx] = ncon((sites[site_idx], isometry, isometry.conj()), ([-1,-2,1,2],[1,-3],[2,-4]))
        
        return MPDO(sites)                                         
    
    def project(self, site_idx, vector): 
        """ contracts with a projector (just a flat isometry) """
        sites = copy.deepcopy(self.sites)
        virt_site = sites.pop(site_idx)
        virt_site = ncon((virt_site, vector, vector.conj()), ([-1,-2,1,2],[1],[2]))

        if self.num_sites == 1: 
            sites = [virt_site]
        elif site_idx == self.num_sites - 1: 
            sites[-1] = ncon((sites[-1], virt_site), ([-1,1,-3,-4],[1,-2]))
        else: 
            sites[site_idx] = ncon((virt_site, sites[site_idx]), ([-1,1],[1,-2,-3,-4]))
        
        return MPDO(sites)
    
    def outcome_prob(self, site_indices, outcomes):
        if type(outcomes) == str: 
            vectors = [] 
            for outcome in outcomes: 
                v = np.zeros(self.local_dim)
                v[int(outcome)] = 1.0
                vectors.append(v)   
        else: 
            vectors = outcomes
            
        offset = 0 
        proj_mpo = copy.deepcopy(self)
        for site_idx, vector in zip(site_indices, vectors):
            site_idx = site_idx - offset
            proj_mpo = proj_mpo.project(site_idx, vector)
            offset += 1
        
        return proj_mpo.trace(scaled=False).real
            
    def diag(self): 
        """ returns MPDO representing diagonal part of the operator """
        sites = copy.deepcopy(self.sites)
        for k in range(self.num_sites):
            site = sites[k]
            s1,s2 = site.shape[0], site.shape[1]
            for i in range(s1):
                for j in range(s2):
                    site[i,j,:,:] = np.diag(np.diag(site[i,j,:,:]))
            sites[k] = site
        return MPDO(sites)
    
    def conj(self): 
        sites = [copy.deepcopy(site).conj().transpose(0,1,3,2) for site in self.sites]
        return type(self)(sites)
    
def contract_sites(site_list): 
    """ takes a list of MPS or MPDO sites and contracts intermediate bonds
    for MPDO sites the two local dimensions are combined into a super local_dim  
    final shape  = (bond_dim, bond_dim, total local_dim) """

    if len(site_list[0].shape) == 4: 
        site_list = [site.reshape(site.shape[0], site.shape[1], site.shape[2] * site.shape[3]) for site in site_list]

    site = site_list[0]
    for i in range(1, len(site_list)): 
        site = ncon((site, site_list[i]), ([-1,1,-3], [1,-2,-4])).reshape(
            site.shape[0], site_list[i].shape[1], site.shape[2]*site_list[i].shape[2])
        
    return site

def contract_mps_env(cur_env, self_site, other_site=None, op=None, right=True):
    if other_site is None:
        other_site = self_site

    if op is None:
        op = np.eye(self_site.shape[-1])

    site = ncon((self_site, op), ([-1,-2,1],[1,-3]))
    s1 = [-2,1,-3] if right else [1,-2,-3]
    s2 = [-2,1,2] if right else [1,-2,2]
    tensor = ncon((cur_env, site), ([1,-1], s1))
    new_env = ncon((tensor, other_site.conj()), ([1,-1,2], s2))
    return new_env

def decompose_site_n(data, msvr=None, max_dim=None, symmetric=False): 
    """ 
    decomposes sites of an MPS or MPDO by combining left_bond and top_spin, and right_bond and bottom_spin
    data.shape = (left_bond, right_bond, top_spin, bottom_spin)
    top_site.shape = (left_bond, new_bond, top_spin)
    bottom_site.shape = (new_bond, right_bond, bottom_spin)
    """
    sh = data.shape
    data = data.transpose(0, 2, 1, 3).reshape(sh[0] * sh[2], sh[1] * sh[3])  
    # shape has become (left_bond * top_spin, right_bond * bottom_spin)

    u,s,vh = svd(data, full_matrices=False, lapack_driver='gesvd')    
    if symmetric:
        u, vh = u @ np.diag(np.sqrt(s)), np.diag(np.sqrt(s)) @ vh
    else: 
        u, vh = u, np.diag(s) @ vh

    if msvr is not None: 
        s = s[s>msvr*s[0]]
    elif max_dim is not None:
        dim = min(max_dim, len(s[s>default_msvr*s[0]]))
        s = s[:dim]
    else: 
        s = s[s>default_msvr*s[0]]

    u = u[:,:len(s)] 
    vh = vh[:len(s),:]

    top_site = u.reshape(sh[0],sh[2],len(s)).transpose(0,2,1)
    bottom_site = vh.reshape(len(s),sh[1],sh[3])
        
    return top_site, bottom_site

def decimate(con_site, local_dim, msvr=None, max_dim=None, symmetric=False): 
    """ decimates sites of an MPS or MPDO after the action of span-local gate 
    con_site.shape = (bond_dim, bond_dim, local_dim**span) """
    span = int(np.log(con_site.shape[2]) / np.log(local_dim))

    dec_sites = []
    for i in range(span-1): 
        con_site = con_site.reshape(con_site.shape[0], con_site.shape[1], local_dim, local_dim**(span-i-1))
        # shape has become (b, b, d, d*d*d*...*d)
        top, con_site = decompose_site_n(con_site, msvr=msvr, max_dim=max_dim, symmetric=symmetric)
        dec_sites.append(top)

    dec_sites.append(con_site)
    return dec_sites

def sv_to_mps(sv, local_dim=2, msvr=None, max_dim=None): 
    """ assumes sv is a vector of shape (local_dim**n,) and attaches fake bonds of dim 1 """
    con_site = sv[np.newaxis, np.newaxis, :]
    dec_sites = decimate(con_site, local_dim, msvr=msvr, max_dim=max_dim, symmetric=False)
    return MPS(dec_sites)

def transfer_matrices(mps1, mps2): 
    tm_list = []
    for site1, site2 in zip(mps1.sites, mps2.sites):
        tm = ncon((site1, site2.conj()), ([-1,-3,1],[-2,-4,1]))
        tm_list.append(tm)
    return tm_list

def left_ortho(site, local_dim=2):
    c = sum([site[:,:,i].conj().T @ site[:,:,i] for i in range(local_dim)])
    return np.allclose(c, np.eye(c.shape[0]), rtol=1e-5, atol=1e-8)

def right_ortho(site, local_dim=2):
    d = sum([site[:,:,i] @ site[:,:,i].conj().T for i in range(local_dim)])
    return np.allclose(d, np.eye(d.shape[0]), rtol=1e-5, atol=1e-8)

def check_center(mps, desired_center, verbose=False):    
    status = []
    for idx, site in enumerate(mps.sites):
        is_left = left_ortho(site, mps.local_dim)
        is_right = right_ortho(site, mps.local_dim)
        
        assert (idx < desired_center and is_left) or (idx > desired_center and is_right) or (idx == desired_center), \
            f"Site {idx} is not properly orthogonalized for center at {desired_center}"
        
        if is_left and is_right:
            status.append('L+R')
        elif is_left:
            status.append('L')
        elif is_right:
            status.append('R')
        else:
            status.append('N')
        
    if verbose: 
        for i, s in enumerate(status):
            print(f"Site {i}: {s}")

    return True

def random_mps(num_sites, max_bond_dim=4, local_dim=2, normalize=True):
    sites = []
    bond_dims = [1] + [max_bond_dim] * (num_sites - 1) + [1]
    
    for i in range(num_sites):
        left_bond_dim = bond_dims[i]
        right_bond_dim = bond_dims[i + 1]
        site = np.random.rand(left_bond_dim, right_bond_dim, local_dim) + 1j * np.random.rand(left_bond_dim, right_bond_dim, local_dim)
        sites.append(site)

    mps = MPS(sites)
    if normalize:
        mps.normalize()
    return mps

def fast_ipr(mps): 
    site0_up = mps[0][0,:,0]
    site0_down = mps[0][0,:,1]

    left_up = ncon([site0_up, site0_up, site0_up.conj(), site0_up.conj()], ([-1], [-2], [-3], [-4]))
    left_down = ncon([site0_down, site0_down, site0_down.conj(), site0_down.conj()], ([-1], [-2], [-3], [-4]))

    left = left_up + left_down

    for i in range(1, n): 
        site_i_up = mps[i][:,:,0]
        site_i_down = mps[i][:,:,1]

        next_list = [] # contract with site_i_up 4 times and add to this list, and then contract with site_i_down 4 times and add to this list, and then we will sum them up
        next_list.append(ncon([left, site_i_up, site_i_up, site_i_up.conj(), site_i_up.conj()], ([1,2,3,4], [1,-1], [2,-2], [3,-3], [4,-4])))
        next_list.append(ncon([left, site_i_down, site_i_down, site_i_down.conj(), site_i_down.conj()], ([1,2,3,4], [1,-1], [2,-2], [3,-3], [4,-4])))
        left = sum(next_list)

    return left[0,0,0,0].real