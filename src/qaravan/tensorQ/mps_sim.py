from qaravan.core import BaseSim, pretty_print_sv
from qaravan.tensorQ import MPS, all_zero_mps, string_to_mps, contract_sites, decimate
from ncon_torch import ncon
import numpy as np

class MPSSim(BaseSim):
    def __init__(self, circ, init_state=None, msvr=None, max_dim=None):
        super().__init__(circ, init_state=init_state, nm=None)    
        self.max_dim = max_dim
        self.msvr = msvr

    def initialize_state(self):
        """ 
        internal state is an MPS with local dimension inherited from the circuit 
        init_state can be provided either as an MPS or a bitstring (and in the future as a statevector)
        """
        if self.init_state is None: 
            self.state = all_zero_mps(self.num_sites, self.local_dim)

        elif type(self.init_state) == MPS: 
            self.state = self.init_state

        elif type(self.init_state) == str:
            self.state = string_to_mps(self.init_state, self.local_dim) 

        else:
            raise ValueError("init_state must be either an MPS or a bitstring.")
        
    def apply_gate(self, gate, symmetric=False):
        inds, mat = gate.indices, gate.matrix 

        if len(inds) > 1 and np.abs(inds[0]-inds[-1]) == 2:
            inds = [inds[0], int(inds[0] + inds[-1])//2, inds[-1]]
            mat = lift_nnn_gate(mat)

        if np.abs(inds[0]-inds[-1]) > 2 or len(inds) > 3:
            raise NotImplementedError("MPS simulator currently only supports up to nn and nnn gates.")

        self.state.canonize(inds[0])
        sites = self.state.sites
        con_site = contract_sites([sites[i] for i in inds])        

        con_site = ncon((mat, con_site), ([-3,1], [-1,-2,1]))
        dec_sites = decimate(con_site, self.local_dim, msvr=self.msvr, max_dim=self.max_dim, symmetric=symmetric)
        self.state.center = inds[-1]
        for i, site in enumerate(dec_sites): 
            sites[inds[i]] = site

    def measure(self, meas_sites):
        raise NotImplementedError("Measurement not yet implemented for MPS simulator.")
    
    def local_expectation(self, local_ops):
        return None
    
    def __str__(self):
        sv = self.state.to_vector()
        return pretty_print_sv(sv, self.local_dim)
    
    def normalize_state(self):
        self.state.normalize() # using MPS normalize method

def lift_nnn_gate(U): 
    d = int(U.shape[0]**0.5)
    return ncon((U.reshape(d,d,d,d), np.eye(d)), ([-1,-3,-4,-6], [-2, -5])).reshape(d**3, d**3)

def measure_mps(mps, meas_sites):
    # inefficient to compute k-RDMs. Instead we do this sequentially. 
    # compute expectation value of projectors on first site, collapse, get new MPS
    # repeat for next site
    # need to first implement local_expectation and then use projectors as operators
    return None