# TODO.md — Qaravan v0.2 rewrite

This file is the active task list for the v0.2 rewrite. Entries are roughly
ordered; each has explicit acceptance criteria. Treat the criteria as the
definition of done.

Before implementing a task review the "Development protocol" section in CLAUDE.md. 

---

## Task 5 — Implement functions to generate common circuits in `core/circuits.py`

**Goal:** Any part of the old circuits.py that didn't end up in base.py should be in this file. Add tests.

**Acceptance:**
- All tests pass.
- No `meas_sites` attribute — terminal measurements should be a method in State output but Simulator. 
---


## Task 6 — Concrete `Observable` subclasses in `core/observables.py`

**Goal:** Implement the observables needed for the cross-check and workflow
tests: `PauliString`, `PauliSum`, `LocalOp`, `Magnetization`. Each has a name, indices and a matrix representation (although this may not be what is used by each state's expectation() method). 

**Tests:**
- `test_pauli_string_as_matrix`: 1-qubit Pauli strings give correct matrices.
- `test_pauli_sum_as_matrix`: sum of Paulis matches direct construction.
- `test_magnetization_on_basis_states`: ⟨Z_i⟩ on basis states.

**Acceptance:**
- tests pass 
---

## Task 7 — Implement first backend: `Statevector` and `StatevectorSimulator`

**Goal:** The reference backend. Everything else gets cross-checked against
this. Located in `backends/statevector.py`. Look at legacy code to understand how this can be done. Note that the contraction code for statevectors, MPS, density matrices, etc have already been worked out in the legacy code and they should NOT be meddled with. Contraction code is VERY easy to get wrong and those codes were written after painstaking hand-written calculations. You should feel free to beautify them, but do not make any changes to the core contraction without asking first.

**Note on Reset/Measure**: `Reset` and `Measure` are not `Gate` subclasses in v0.2.
They are methods on concrete `State` subclasses. Implement `Statevector.reset(sites, reset_to)` and `Statevector.measure(sites)` here (the latter is equivalent to `sample_and_collapse`). This replaces the legacy `Reset(Gate)` class.

**Tests first (the hard set):**
- `test_statevector_init_bitstring`: `Statevector(bitstring="01")` has the
  correct array.
- `test_statevector_init_random_seed`: reproducible with same seed.
- `test_statevector_norm`: random Statevectors are normalized.
- `test_statevector_sim_ghz`: building and running the GHZ circuit gives
  `(|000⟩ + |111⟩)/√2`.
- `test_statevector_expectation_pauli`: `<Z>` on `|0⟩` is 1; `<X>` on `|+⟩`
  is 1; `<ZZ>` on Bell state is 1.
- `test_statevector_sim_raises_on_noise`: passing any noise model raises
  `IncompatibleNoiseError`.
- `test_statevector_sample_and_collapse`: outcome statistics match Born
  rule over many shots; collapsed state is correct.

**Acceptance:**
- All tests pass.
- The workflow examples in CLAUDE.md that involve statevector section directly run as
  written.

---

## Task 8 — Concrete `NoiseModel` subclasses in `core/noise.py`

**Goal:** Port `ThermalNoise`, `PauliNoise`, `PauliLindbladNoise` to the new
ABC. 

**Tests:**
- `test_thermal_noise_superoperator_cptp`: the superoperator is CPTP.
- `test_pauli_noise_as_pauli_channel`: sums to probability 1.
- `test_thermal_noise_as_pauli_channel_raises`: clear error message.

---

## Task 9 — Implement second backend: `DensityMatrix` and `DensityMatrixSimulator`

**Goal:** Second backend. First test of the cross-check infrastructure.
First backend that accepts a `NoiseModel`. Located in
`backends/density_matrix.py`. Look at legacy code to understand how this can be done. Note that the contraction code for statevectors, MPS, density matrices, etc have already been worked out in the legacy code and they should NOT be meddled with. Contraction code is VERY easy to get wrong and those codes were written after painstaking hand-written calculations. You should feel free to beautify them, but do not make any changes to the core contraction without asking first. 


**Tests first:**
- `test_density_matrix_init_from_bitstring`.
- `test_density_matrix_trace_one`.
- `test_dm_sim_unitary_agrees_with_sv`: cross-check against
  `StatevectorSimulator` for a noiseless circuit. This is the test that
  earns DM's place in the codebase.
- `test_dm_sim_multi_site_expectation`: explicit test for the v0.1 regression
  — `<Z_0 Z_1>` on a Bell state computed by DM matches the SV result and the
  known value of 1. (This test is the deliverable that proves we did not
  reintroduce the v0.1 bug.)
- `test_dm_sim_with_thermal_noise`: a circuit run with ThermalNoise produces
  a state with trace 1 and the expected decohered diagonal.

**Acceptance:**
- All tests above pass.
---


## Task 10 — First Hamiltonian: `TFI` in `core/hamiltonians.py`

**Goal:** Port TFI to the new design. It produces `Circuit`s and
`Observable`s; it does not subclass any ABC.

**Tests:**
- `test_tfi_ground_state_energy_h0`: at $h=0$, ground state is product of
  $|0\rangle$, energy is $-J(n-1)$.
- `test_tfi_ground_state_against_exact_diag`: small $n$, compare to
  numpy.linalg.eigh.
- `test_tfi_trotter_circuit_against_propagator`: 2nd-order Trotter at small
  $dt$ approximates `expm(-iHt)` to expected accuracy.

---

## Task 11 — First end-to-end workflow: noisy Trotter sweep

**Goal:** The workflow in CLAUDE.md "Noisy Trotter evolution with an
observable" runs end-to-end. Acceptance test for phases 0-2 of the v0.2
rewrite.

**Acceptance:**
- The code block in CLAUDE.md runs exactly as written.
- Output magnetization curve matches v0.1 output (legacy reference) within
  numerical tolerance.

---

## Tasks 12+: backends, applications, and research projects

After tasks 1-11, the spine is in place. Subsequent tasks are added as needed:

- MPS backend + MPS-vs-SV cross-checks.
- MPDO backend + MPDO-vs-DM cross-checks.
- Mid-circuit measurements as first-class (`Measure`, conditional gates,
  classical register schema on `Circuit`).
- Monte Carlo backend with `Walkers`.
- Clifford backend (wraps Stim).
- Matchgate backend (with non-Gaussian extension on the roadmap).
- Pauli propagation backend (wraps `PauliPropagation.jl`).
- Variational state prep workflow + brickwall ansatze + RunContext port.
- AKLT prep + string order workflow.
- QCP / absorbing-state-calibration workflow.
- Harness shots-dataframe ingestion.

Each will get its own proposal and acceptance criteria when promoted to
active.

---