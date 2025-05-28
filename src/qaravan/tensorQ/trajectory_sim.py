from .statevector_sim import *

class TrajectorySim(BaseSim):
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

class PECSim(TrajectorySim): 
    """ simulates probabilistic error cancellation via trajectories """
    def __init__(self, circ, init_state=None, nm=None):
        super().__init__(circ, init_state=init_state, nm=nm)
    
    def circuit_sample(self):
        circ = copy.deepcopy(self.circ)
        if not circ.built: 
            circ.decompose()
            circ.construct_layers()
        
        if self.nm is not None: 
            overall_prefac = 1.0
            layers = []
            for layer in circ.layers:
                noise_layer = self.nm.sample()
                correction_layer, prefac = self.nm.inv_sample()
                overall_prefac *= prefac
                layers.extend([layer, noise_layer, correction_layer])
            circ.layers = layers
        
        circ.built = True
        return circ, overall_prefac
    
    def expectation_sample(self, obs):
        circ, overall_prefac = self.circuit_sample()
        sim = StatevectorSim(circ, init_state=self.init_state)
        sim.run(progress_bar=False)
        sv = sim.get_statevector()
        expectation_value = np.vdot(sv, obs @ sv).real
        return overall_prefac * expectation_value