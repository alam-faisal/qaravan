from qaravan.core import BaseSim, pretty_print_sv
from qaravan.tensorQ import MPS, all_zero_mps, string_to_mps, contract_sites, decimate
from ncon_torch import ncon

class MPSSim(BaseSim):
    def __init__(self, circ, init_state=None):
        super().__init__(circ, init_state=init_state, nm=None)    

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
        
    def apply_gate(self, gate):
        inds, mat = gate.indices, gate.matrix
        self.state.canonize(inds)
        sites = self.state.sites
        con_site = contract_sites([sites[i] for i in inds])        
        con_site = ncon((mat, con_site), ([-3,1], [-1,-2,1]))
        dec_sites = decimate(con_site, self.local_dim)
        for i, site in enumerate(dec_sites): 
            sites[inds[i]] = site

    def measure(self, meas_sites):
        raise NotImplementedError("Measurement not yet implemented for MPS simulator.")
    
    def local_expectation(self, local_ops):
        return None
    
    def __str__(self):
        sv = self.state.to_vector()
        return pretty_print_sv(sv, self.local_dim)