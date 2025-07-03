from .statevector_sim import *

class MonteCarloSim(BaseSim):
    """ 
    noisy simulation for channels that can be thought of as 
    a pseudo-probability distribution over unitaries applied to statevector
    nm must have a method 'sample' that returns a list of Gates
    main output of simulator is expectation values
    unlike other simulators it doesn't store internal states  
    """   
    def __init__(self, circ, init_state=None, nm=None):
        super().__init__(circ, init_state=init_state, nm=nm)  

    def circuit_sample(self): 
        circ = copy.deepcopy(self.circ)
        if not circ.built: 
            circ.decompose()
            circ.construct_layers()
        
        if self.nm is not None: 
            noise_layers = [self.nm.sample() for _ in range(len(circ.layers))]
            circ.layers = [
                circ.layers[i//2] if i % 2 == 0 else noise_layers[i//2]
                for i in range(len(circ.layers) * 2)
            ]

        circ.built = True
        return circ
    
    def expectation_sample(self, obs): 
        """ for now obs is a matrix of correct dimension """
        circ = self.circuit_sample()
        sim = StatevectorSim(circ, init_state=self.init_state)
        sim.run(progress_bar=False)
        sv = sim.get_statevector()
        return np.vdot(sv, obs @ sv).real