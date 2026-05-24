# CLAUDE.md — Qaravan

Qaravan is a Python library for classical simulation of quantum circuits. It is the research playground for my work on near-term quantum algorithms, error mitigation, and mid-circuit measurement applications. 

The guiding principle for Qaravan is that it should enable **lightning-fast iteration** of ideas. Performance matters, but ergonomics, discoverability, and trustworthiness of results matter more. When making design choices, prefer the option that lets an existing preprint or a new paper idea be prototyped within an hour of coding. 

Qaravan is currently undergoing a complete overhaul with a new clean abstraction that facilitates this guiding principle. 

---

## Current repo layout

```
qaravan/
  core/         # Circuit, Gate, Hamiltonian, Pauli toolkit, noise models, lattices
  tensorQ/      # StatevectorSim, DensityMatrixSim, MPSSim, MonteCarloSim and base structures for tensors 
  algebraQ/     # MatchgateSim, CliffordSim, g-sim 
  applications/ # Higher-level helpers: Trotter sweeps, state prep, magnetization
tests/          # unit test suite — run with `pytest tests/` from repo root before every commit
examples/       # Notebooks demonstrating and verifying canonical workflows; run after major changes
```

## Target repo layout

```
qaravan/
  core/
    base.py           # ABCs: State, Gate, Circuit, Simulator, Observable, NoiseModel
    gates.py          # Concrete gates: H, X, Y, Z, CNOT, RX/Y/Z, RXX/YY/ZZ, Reset, Measure
    circuits.py       # functions for generating common circuits
    observables.py    # Concrete observables: Pauli strings, magnetization, etc 
    noise.py          # ThermalNoise, PauliNoise, PauliLindbladNoise
    hamiltonians.py   # TFI, XXZ, Heisenberg, output is to usually produce Circuit objects
    lattices.py       # Linear, square, toric, Kagome, output is to usually produce Hamiltonians or Circuits
  backends/
    statevector.py    # Statevector(State) + StatevectorSimulator(Simulator)
    density_matrix.py # DensityMatrix(State) + DensityMatrixSimulator(Simulator)
    mps.py            # MPS(State) + MPSSimulator(Simulator)
    mpdo.py           # MPDO(State) + MPDOSimulator(Simulator)
    _tn_internals.py  # code for tensor contraction used by mps.py and mpdo.py
    monte_carlo.py    # Walkers(State) + MonteCarloSimulator(Simulator)
    clifford.py       # StabilizerTableau(State) + CliffordSimulator(Simulator); wraps Stim
    matchgate.py      # GaussianState(State) + MatchgateSimulator(Simulator); 
    pauli_propagation.py  # PauliSum(State) + PauliPropagationSimulator(Simulator); wraps PauliPropagation.jl
  applications/       # Higher-level helpers for Trotter sweeps, compilation, dynamic quantum circuits
tests/                # unit test suite — run with `pytest tests/` from repo root before every commit
examples/             # Notebooks demonstrating and verifying canonical workflows; run after major changes
```

The `core/base.py` file is the design document. All ABCs live there together so the architecture is legible at a glance. Concrete State and Simulator subclasses are co-located in one file per backend, because they evolve in lockstep. Backends with substantial internal machinery (MPS, MPDO) place implementation details in `_internals.py` files; the public API always imports from the backend's top-level file. The `tensorQ`/`algebraQ` distinction from v0.1 is dropped in v0.2. 


## Core abstractions

The design rests on four main classes (`Gate`, `State`, `Observable`, `NoiseModel`) plus one top-level orchestrator (`Simulator`) and one connective tissue (`Circuit`). 

### `Gate`

A `Gate` has a `name`, `indices` (which sites it acts on), and a **default representation** as a $d^k \times d^k$ matrix for a $k$-site gate on local dimension $d$. Non-unitary operations (`Reset`, `Measure`) are also considered to be `Gate` subclasses — the abstraction does not distinguish unitary from non-unitary at the type level. Gates do *not* know about backends; translation to the natural representation for a backend is done at *compile* time and is the Simulator's job.

### `Circuit`

A `Circuit` is a sequence of `Gate` objects, nothing more. It supports composition (`+`), repetition (`*`), and `dagger()`, and has methods for looking up its properties. It does not carry simulator-specific data. How it handles parametrization will be dealt with in the future. 

### `State`

A `State` carries the data structure a particular simulator evolves. Subclasses include `Statevector`, `DensityMatrix`, `MPS`, `MPDO`, `StabilizerTableau`, `PauliSum`, and `Walkers` (a collection of underlying states). The semantic content of a `State` is not always a physical quantum state — `PauliSum` in Pauli propagation represents a Heisenberg-picture observable, for instance. This is a feature: the abstraction tracks *what evolves under the circuit*, not *what physicists call a state*.

**Key invariant:** once a `Simulator` returns a `State`, that `State` is independent of the simulator and the circuit. This enables dynamic-circuit workflows where a state is fed back into a new simulator instance.

State subclasses own operations like: `state.expectation(observable)`, `state.sample(shots)`, `state.sample_and_collapse(sites)`, `state.rdm(sites)`, `state.overlap(other_state)`. Simulators do not have these methods. Conversions between state types are explicit and named (`statevector.to_mps(max_chi=...)`), never silent. Each subclass also provides methods to generate common states (common to that state's backend). 

### `Simulator`

A `Simulator` takes an initial `State`, a `Circuit`, and an optional `NoiseModel`. It **compiles** the circuit into a backend-specific form, evolves the state, and returns the final `State`. That is all. The compile step is a pipeline with three stages:

1. **Noise insertion** — backend-specific (e.g., density matrix adds idling-channel layers; Monte Carlo samples Pauli operators between gates).
2. **Gate translation** — each gate is converted to the backend's natural representation via `sim.translate_gate(gate)`. Translation lives on the simulator, not the gate, because simulators outnumber gates and translation can fail in backend-specific ways. Translation failures raise clear, named errors (e.g., `MatchgateSim.translate_gate(non_matchgate)` raises `NotMatchgateError`); they never silently produce wrong answers. These natural representations can be matrices meant to act on pure states (statevector and MPS backends), superoperators, SO(2n) matrices, operations on StabilizerTableau or PauliSum. 

3. **Optional optimization** — layer fusion, MPS gate truncation, etc.

Simulator constructors carry backend-specific options (`max_chi`, `truncation_threshold`). Validating the initial state type is the simulator's responsibility; incompatible types raise at construction.

### `Observable`

An `Observable` is parallel to `Gate`: it has a name, index and a default representation (typically $d^k \times d^k$ Hermitian matrix) and methods to produce backend-specific forms. The expected forms are `as_pauli_sum()`, `as_matrix()`, `as_local_ops(sites)`, and `as_majorana_polynomial()`. A `State` subclass asks the `Observable` for whichever representation it needs to compute the expectation efficiently.

### `NoiseModel`

Each simulator asks the passed `NoiseModel` to provide a natural representation. This is either superoperator for density matrix and MPDO simulators or sampled Pauli strings for MonteCarlo or Clifford simulators. Note that not every subclass of `NoiseModel` need have every method (it's hard to sample Pauli strings from ThermalNoise, for instance). 

### Others

Hamiltonians (TFI, XXZ, Heisenberg, ...) and lattices are *physics objects*, not part of the ABC hierarchy. Their job is to produce `Circuit`s (via `H.trotter_circ(...)`) and `Observable`s (via `H.as_observable()`). Once that's done, the simulator and state abstractions take over and the Hamiltonian is no longer relevant to the simulation code.


## Coding style (Faisal's preferences — please follow)

- **Type hints + good variable names beat long docstrings.** Use them consistently in public APIs.
- **Docstrings are one or two lines plus non-obvious notes.** No boilerplate `Args:` blocks that restate type hints. Include non-trivial shapes, axis ordering and other things users calling and developing on the function might care about. 
- **Functional by default; classes only when state genuinely warrants it.** `Circuit`, `Gate`, simulators, `Hamiltonian`, `MPS`: yes. Random utilities: no.
- **Short functions.** If a function is over ~40 lines, consider whether it's really one thing.
- **`uv` for package management.** Not pip, not conda.
- **Use outside-in test driven development** first write structural scaffold, then write extensive tests for correctness of physics and logic, and then write the code to satisfy the tests

---

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