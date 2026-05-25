
## Expected workflows upon completion of v0.2 

After v0.2 is completely developed, we should be able to do at least a large chunk of the following:


### Simulate a circuit in different simulators

```python
circ = Circuit([H(0), CNOT([0, 1])])
obs = PauliString("IX")

thermal_nm = ThermalNoise(t1=100, t2=75, t_q1=5, t_q2=20)
pauli_nm = PauliNoise([PauliString("II"), PauliString("ZZ")], probs=[0.9, 0.1])

initial_sv = Statevector(bitstring="01")
final_sv = StatevectorSimulator(circ, initial_sv).run()
exp = final_sv.expectation(obs)

initial_dm = DensityMatrix(bitstring="01")
final_dm = DensityMatrixSimulator(circ, initial_dm, thermal_nm).run()
exp = final_dm.expectation(obs)

initial_mps = MPS(bitstring="01")
final_mps = MPSSimulator(circ, initial_mps).run()
exp = final_mps.expectation(obs)

initial_mpdo = MPDO(bitstring="01")
final_mpdo = MPDOSimulator(circ, initial_mpdo, thermal_nm).run()
exp = final_mpdo.expectation(obs)

initial_tableau = StabilizerTableau(bitstring="01")
final_tableau = CliffordSimulator(circ, initial_tableau).run()
exp = final_tableau.expectation(obs)

initial_gaussian = GaussianState(bitstring="01")
final_gaussian = MatchgateSimulator(circ, initial_gaussian).run()
exp = final_gaussian.expectation(obs)

initial_ps = PauliSum(obs) 
final_ps = PauliPropagationSimulator(circ, initial_ps).run()
exp = final_ps.expectation(PauliSum(bitstring="01"))

initial_walkers = Walkers(bitstring="01", pure_state_type=Statevector, num_walkers=1000).run()
final_walkers = MonteCarloSimulator(circ, initial_walkers, pauli_nm).run()
exp = final_walkers.expectation(obs)
```

### Simulate dynamic quantum circuits

```python 
sv = Statevector(bitstring="01")
meas_sites = [1]

def decoder(outcome, sv, round) -> Circuit: 
  return Circuit() 

for round in range(n_rounds): 
  outcome, sv = sv.measure_and_collapse(meas_sites)
  circ = decoder(outcome, sv, round)
  sv = StatevectorSimulator(circ, sv).run()
```

### Noisy AKLT prep + string-order measurement

```python 
circ = aklt_prep_circuit(n) 
obs = StringOrderObservable(indices=range(n), op_type="Z")

initial_dm = DensityMatrix(bitstring='0'*n)
final_dm = DensityMatrixSimulator(circ, initial_dm).run()
exp = final_dm.expectation(obs)
```

### Noisy Trotter evolution with an observable

```python

ham = TFI(n, jz=1.0, h=0.75)
nm = QubitNoise(t1=100, t2=75, t_q1=0.04, t_q2=0.5)
obs = Magnetization(n)
exp_list = trotter_dm_sim(ham, step_size=0.1, max_steps=500,
                          nm=nm, obs=obs)

def trotter_dm_sim(ham, step_size, max_steps, nm, obs) -> np.ndarray: 
  circ = ham.trotter_circuit(step_size)
  dm = DensityMatrix(bitstring='0'*ham.num_sites)
  exp_list = []
  for step in range(max_steps): 
    exp_list.append(dm.expectation(obs))
    dm = DensityMatrixSimulator(circ, dm, nm).run()

  return exp_list                        
```

### Variational state preparation (environment sweep)

```python

target_sv = Statevector(n, random_seed=42)
skeleton = brickwall_skeleton(n=4, depth=2)
context = RunContext(max_iter=10000, stop_ratio=1e-8, stop_absolute=1e-7)
circ, cost_list = environment_state_prep(target_sv, skeleton=skeleton,
                                         context=context)
```

---

## Scope of Qaravan

**In scope:**
- Mid-circuit measurements and feed-forward (this is a research pillar — make it easy to code up ergonomic)
- Structure-aware noise and error mitigation techniques
- Classical simulation backends (tensor networks, matchgate, Clifford, Pauli propagation)
- Commonly used Hamiltonians and observables for canonical near-term experiments
- Variational state prep and circuit synthesis tools
- Being able to deal with shots dataframes output by Phasecraft's Harness infrastructure to test error mitigation techniques. 

**Out of scope (for now):**
- Linking to actual quantum hardware. Phasecraft's pipeline handles device submission. Qaravan stays classical.
- Heavy GPU optimization beyond the existing torch backend.
- Compiling to or from external circuit formats (Qiskit, Cirq). If a conversion is needed, do it ad hoc in a notebook, not in the library.

---