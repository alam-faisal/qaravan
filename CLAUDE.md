# CLAUDE.md — Qaravan

Qaravan is a Python library for classical simulation of quantum circuits. It is the research playground for my work on near-term quantum algorithms, error mitigation, and mid-circuit measurement applications. 

The guiding principle for Qaravan is that it should enable **lightning-fast iteration** of ideas. Performance matters, but ergonomics, discoverability, and trustworthiness of results matter more. When making design choices, prefer the option that lets an existing preprint or a new paper idea be prototyped within an hour of coding. 

Qaravan is currently undergoing a complete overhaul with a new clean abstraction that facilitates this guiding principle. 

---


## Repo layout

```
qaravan/
  legacy/             # Old code that is very useful to look at for the underlying logic
  core/
    base.py           # ABCs: State, Gate, Simulator, Observable, NoiseModel
    gates.py          # Concrete gates: H, X, Y, Z, CNOT, RX/Y/Z, RXX/YY/ZZ, Reset, Measure
    circuits.py       # Circuit class and functions for generating common instances
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

The `core/base.py` file is the design document. All ABCs live there together so the architecture is legible at a glance. Concrete State and Simulator subclasses are co-located in one file per backend, because they should be developed together. Backends with substantial internal machinery (MPS, MPDO) place implementation details in `_internals.py` files.


## Core abstractions

The four main classes are (`Gate`, `State`, `Observable`, `NoiseModel`). There is also a top-level orchestrator (`Simulator`) and a container class (`Circuit`). 

### `Gate`

A `Gate` has a `name`, `indices` (which sites it acts on), and a **default representation** as a $d^k \times d^k$ matrix for a $k$-site gate on local dimension $d$. Non-unitary operations (`Reset`, `Measure`) are also considered to be `Gate` subclasses. Gates do *not* know about backends; translation to the natural representation for a backend is done at *compile* time and is the Simulator's job.

### `Circuit`

A `Circuit` is a sequence of `Gate` objects. It supports composition (`+`), repetition (`*`), indexing, slicing and `dagger()`. It uses `decompose` to break gates into some native gateset (if asked) and `construct_layers` to break the sequence of gates into layers, (currently assumes unlimited entangling zones). It does not carry simulator-specific data. How it handles parametrization will be dealt with in the future. 

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

## API vision 

Whenever you need to know what the API should look like so you can develop the source code appropriately, load docs/API_VISION.md and find a relevant example. 


## Coding style (Faisal's preferences — please follow)

- **Type hints + good variable names beat long docstrings.** Use them consistently in public APIs.
- **Docstrings are one or two lines plus non-obvious notes.** No boilerplate `Args:` blocks that restate type hints. Include non-trivial shapes, axis ordering and other things users calling and developing on the function might care about. 
- **Functional by default; classes only when state genuinely warrants it.** `Circuit`, `Gate`, simulators, `Hamiltonian`, `MPS`: yes. Random utilities: no.
- **Short functions.** If a function is over ~40 lines, consider whether it's really one thing.
- **`uv` for package management.** Not pip, not conda.
- **Use outside-in test driven development** first write structural scaffold, then write extensive tests for correctness of physics and logic, and then write the code to satisfy the tests

## Development protocol

This section describes the protocol for collaborating on Qaravan. Every step matters. If
something in this protocol conflicts with a faster path, follow the protocol.

### Before starting a task

1. Pick a task from TODO.md (either the topmost or something specified by user) 

2. **Write a proposal.** Drop a markdown file in `proposals/YYYY-MM-DD-task-name.md` concisely describing:
   - The approach.
   - Classes, methods, and functions you plan to write, with constructors and signatures.
   - How you plan to handle complications and edge cases.
   - Anything you are unsure about and want the user's opinion.
   In the proposal phase feel free to read files and do whatever necessary to understand the task but do NOT modify code under src. 

3. **Wait for explicit approval** from the user before writing any implementation code. 

### During a task

4. **Break the task into sub-tasks.** After every few sub-tasks, when there's
   a logical checkpoint:
   - Run the relevant tests.
   - `git add` the specific files changed (not `git add .`).
   - `git commit` with a small, atomic commit message describing exactly that
     sub-task.

   Each task may end up containing several commits, one per logical change.

5. **When something goes wrong, stop and discuss it.** A failing test that
   reveals a design flaw, an edge case the proposal didn't anticipate, a
   question that needs the user's input — write up what you've found in the
   report (see step 7) and ask. Do not push through with a workaround that
   wasn't in the approved proposal.

### Finishing a task

6. **Write an example notebook.** Add a nicely named `.ipynb` under
   `examples/devbooks/` walking the user through the newly added features —
   specifically the ones the user should familiarize themselves with and
   check by hand. Some content may overlap with tests you wrote, but the
   notebook should focus on *physics correctness* checks that unit tests
   either do or do not capture.

7. **Write a report** to `reports/YYYY-MM-DD-task-name.md` containing:
   - Whether the task succeeded or failed.
   - On success: a short summary of what was accomplished, including
     signatures of new functions and classes.
   - On failure: a clear description of the bug or open question blocking
     progress.
   - A checklist of things the user should verify by hand: notebooks to
     open, tests to read, READMEs to review.

8. **Wait for explicit user confirmation** that they've reviewed the report
   and are satisfied. Then:
   - Remove the task from `TODO.md`.
   - Run the full unit test suite (`uv run pytest tests/`).
   - Run `uv run ruff format src/ tests/` then `uv run ruff check src/ tests/`.
   - `git push` to GitHub. This is the **only** push event in the workflow —
     individual commits stay local until the task is complete and approved.

### End of day

9. **On the user's signal to wrap up**, append an entry to `DAY.md` containing:
   - The date.
   - A concise list of what was accomplished that day.
   - A list of what's up for the next day: anything still awaiting the user's
     review, the next task on the docket, open questions, etc.