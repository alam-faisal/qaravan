from tqdm import tqdm

class BaseSim:
    """ 
    accepts Circuit object and Noise object 
    different simulators use different representations of quantum state
    and different methods for applying gates and measurements
    """

    def __init__(self, circ, init_state=None, nm=None):
        self.circ = circ
        self.num_sites = circ.num_sites
        self.local_dim = circ.local_dim
        self.nm = nm
        self.init_state = init_state
        self.state = None
        self.ran = False

    def initialize_state(self):
        raise NotImplementedError("must be implemented by child classes.")

    def apply_gate(self, gate):
        raise NotImplementedError("must be implemented by child classes.")

    def run(self, progress_bar=True, debug=0):
        """
        progress_bar: If True, display a progress bar during simulation.
        debug: If 1, prints intermediate state after each layer.
        debug: If 2, prints intermediate state after each gate.
        """
        self.initialize_state()
        circ = self.circ.build(self.nm) if not self.circ.built else self.circ
        layers = tqdm(circ.layers) if progress_bar else circ.layers

        for layer in layers:
            for gate in layer:
                self.apply_gate(gate)
                if debug == 2:
                    print(self)
            
            if debug == 1:
                print(self)

        if circ.meas_sites is not None:
            self.measure(circ.meas_sites)

        self.ran = True
        return self.state
    
    def measure(self, meas_sites):
        raise NotImplementedError("must be implemented by child classes.")

    def __str__(self):
        """ pretty prints the state """
        raise NotImplementedError("must be implemented by child classes.")