# TODO.md — Qaravan v0.2 rewrite

Active task list for the v0.2 rewrite. Entries are roughly ordered; each has explicit
acceptance criteria. The version number in the section title is what the project bumps to
**once everything in that section is done**.

Before implementing a task review the "Development protocol" and "Coding style" sections
in CLAUDE.md.

---

## v0.2.2 — Track B: Hamiltonians and Fermions + Track A: Dynamic Quantum Circuits

---

### Task 20 — Generic fermion-to-qubit encoding: `FermionOperator` + Jordan-Wigner transform

**Top priority — do this before Task 14.**

**Goal:** Decouple fermionic model definitions from qubit encoding. `FermiHubbard1D` currently
hand-derives its Jordan-Wigner Pauli strings per `ordering` inside `_build_fh_terms`
(`core/hamiltonians.py`) — correct (see `docs/jw_fermi_hubbard_1d.md`), but it means
re-deriving JW by hand for every new fermionic model, and swapping encodings (e.g.
Bravyi-Kitaev) is a rewrite rather than a parameter. Introduce a small fermionic-operator
algebra and a generic encoding transform that any fermionic Hamiltonian (present and future —
see Task 21) can go through.

**New module `core/fermions.py`:**
- `FermionTerm` — `coeff: complex`, `ops: list[tuple[int, bool]]` (mode index, is_dagger);
  left-to-right operator order is preserved (no auto normal-ordering — multiplication order
  matters and is the caller's responsibility, matching how the term is physically written).
- `FermionOperator` — list of `FermionTerm` + `num_modes`; `+`, scalar `*`, `dagger()`.
- `c(mode)`, `cdag(mode)` — single-ladder-operator `FermionTerm` builders.
- `jordan_wigner(terms: list[FermionTerm], num_modes: int, mode_order: list[int] | None = None) -> list[HamiltonianTerm]`
  — `mode_order[qubit] = mode`, defaults to identity. One `HamiltonianTerm` per input
  `FermionTerm`, embedded on the qubits it actually touches (JW strings are contiguous by
  construction). `ordering` choices (stacked/interleaved/anything else) become a `mode_order`
  permutation, not a code branch.

**New primitive in `core/observables.py`:** `PauliString`/`PauliSum` operator composition
(matrix product in the shared qubit space, not tensor product) — needed to multiply
per-ladder-operator Pauli representations together in order. General-purpose beyond fermions.

**Refactor:** `_build_fh_terms` rewritten to (1) build the FH `FermionOperator` once in a
canonical mode order, (2) compute the `stacked`/`interleaved` `mode_order` permutation, (3)
call `jordan_wigner`. Trotter grouping (even/odd bonds, periodic boundary terms) is
model-specific and stays in `hamiltonians.py`.

**Acceptance:**
- All existing `test_fermi_hubbard.py` tests pass unchanged (same public API, same numerics)
  — the regression safety net.
- New test: `jordan_wigner` reproduces the hand-derived Pauli strings in
  `docs/jw_fermi_hubbard_1d.md` eq. (8)-(10) and (15)-(17) exactly, for both orderings.
- New test: generic Pauli composition reproduces known identities (e.g. `X @ Y == iZ`) and is
  associative.

---

### Task 21 — Anderson impurity model

**Do immediately after Task 20**, as the first real test that the generic encoding
generalizes beyond a 1D chain.

**Goal:** Single-impurity Anderson model, built on Task 20's `FermionOperator`/
`jordan_wigner`: `H = ε_d n_d↑ + ε_d n_d↓ + U n_d↑ n_d↓ + Σ_k ε_k n_k + Σ_k V_k (c†_d c_k +
h.c.)` — a star topology (impurity coupled to every bath mode), unlike FH's nearest-neighbor
chain. JW strings between the impurity and each bath mode will generally have non-trivial
length depending on `mode_order`.

**Acceptance:** TBD in proposal — at minimum, the free-fermion limit (`U=0`) matches exact
diagonalization of the quadratic form, and the atomic limit (`V_k=0`) reproduces the isolated
impurity spectrum.

---

### Task 14 — Environment sweep for DQCs

**Goal:** Extend `environment_state_prep` to handle `DynamicCircuit` targets: optimise the
pre-measurement circuit via environment updates, holding the decoder fixed. Single-round
measurement only; decoder is a fixed lookup table.

**Background — how the environment changes:**

For a DQC with one round of measurements at sites S, and target state `|target⟩` (the
desired final state for each measurement outcome m):

The cost is the average infidelity over Born-weighted outcomes:

```
C = 1 - Σ_m p_m * |⟨target_m | U_post_m P_m U_pre | init⟩|
```

where `p_m = |P_m U_pre|init⟩|²` is the Born probability of outcome m, `P_m` is the projector
onto outcome m, and `U_post_m` is the (fixed) decoder circuit for outcome m.

For pre-circuit gate k, the effective environment is:

```
E_k = Σ_m (1/√p_m) * partial_overlap(pre_state_m, post_state_m, skip=indices_k)
```

where:
- `pre_state_m = G_{k-1}...G_0 |init⟩` (incremental, same as single-state)
- `post_state_m = G_{k+1}†...U_pre† P_m† U_post_m† |target_m⟩` (one per outcome)

The `1/√p_m` weight comes from the fact that the projected state `P_m U_pre|init⟩` has norm
`√p_m`, which must be divided out before computing the overlap with `|target_m⟩`. The Julia
reference code handles this as `inner(mps, t_mps)/renorm` where `renorm = √p_m`.

The key subtlety vs. multi-state prep: `p_m` depends on the current pre-circuit and changes
as gates are updated. In the Julia code this is handled by recomputing `p_m` once per sweep
(not per gate), which is a consistent and stable approximation.

**Functions:**
- `dqc_environment_state_prep(target_ensemble: list[State], dqc: DynamicCircuit, context=None)`
  — returns `(optimised_DynamicCircuit, cost_list)`. `target_ensemble[m]` is the target state
  for measurement outcome m; ordering must match `measure_and_collapse` outcome indexing.
- Private helper `_dqc_environment_update(...)` — same interface as `_environment_update` but
  sums over outcomes with Born weights.

**Do NOT start this task without the Julia code as reference** — the normalisation subtleties
are easy to get wrong. The Julia code is in `legacy/julia/train_dqc.jl` (or wherever it ends up).

**Acceptance:** Notebook shows `dqc_environment_state_prep` learning the correct decoder
circuit for GHZ-from-clusters for a 4-qubit target.

---

## v0.2.3 — Noise and Density Matrix

---

### Task 8 — Concrete `NoiseModel` subclasses in `core/noise.py`

**Goal:** Port `ThermalNoise`, `PauliNoise`, `PauliLindbladNoise` to the new ABC.

**Tests:**
- `test_thermal_noise_superoperator_cptp`: the superoperator is CPTP.
- `test_pauli_noise_as_pauli_channel`: sums to probability 1.
- `test_thermal_noise_as_pauli_channel_raises`: clear error message.

---

### Task 9 — Implement second backend: `DensityMatrix` and `DensityMatrixSimulator`

**Goal:** Second backend. First test of the cross-check infrastructure.
First backend that accepts a `NoiseModel`. Located in
`backends/density_matrix.py`. Look at legacy code to understand how this can be done. Note
that the contraction code for statevectors, MPS, density matrices, etc have already been
worked out in the legacy code and they should NOT be meddled with. Contraction code is VERY
easy to get wrong and those codes were written after painstaking hand-written calculations.
You should feel free to beautify them, but do not make any changes to the core contraction
without asking first.

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

**Acceptance:** All tests above pass.

---

### Task 11 — First end-to-end workflow: noisy Trotter sweep

**Goal:** The workflow in CLAUDE.md "Noisy Trotter evolution with an observable" runs end-to-end.
Acceptance test for phases 0-2 of the v0.2 rewrite.

**Acceptance:**
- The code block in CLAUDE.md runs exactly as written.
- Output magnetization curve matches v0.1 output (legacy reference) within numerical tolerance.

---

## v0.2.4 — Tensor Network Backends

---

### Task: MPS backend + MPS-vs-SV cross-checks

Implement `MPS(State)` and `MPSSimulator(Simulator)` in `backends/mps.py`.
Cross-check against `StatevectorSimulator` for noiseless circuits.

---

### Task: MPDO backend + MPDO-vs-DM cross-checks

Implement `MPDO(State)` and `MPDOSimulator(Simulator)` in `backends/mpdo.py`.
Cross-check against `DensityMatrixSimulator` for noisy circuits.

---

## v0.2.5 — Specialised Simulators

---

### Task: Monte Carlo backend extensions

Extend `Walkers` and `MonteCarloSimulator` (implemented in v0.2.2 Task 19) with
mid-circuit measurement support and the full `PauliLindbladNoise` sampler.

---

### Task: Clifford backend (wraps Stim)

Implement `StabilizerTableau(State)` and `CliffordSimulator(Simulator)` in `backends/clifford.py`.

---

### Task: Matchgate backend

Implement `GaussianState(State)` and `MatchgateSimulator(Simulator)` in `backends/matchgate.py`.
Non-Gaussian extension on the roadmap.

---

### Task: Pauli Propagation backend (wraps PauliPropagation.jl)

Implement `PauliSum(State)` and `PauliPropagationSimulator(Simulator)` in
`backends/pauli_propagation.py`.

---

## v0.2.6 — Applications: reproducing paper results

---

### Task: Multi-state environment sweep

**Goal:** Compile a fixed-structure circuit mapping `|init_m⟩ → |target_m⟩` for m = 1…M
simultaneously. Physical use cases: quantum channel compilation, scrambler verification,
isometry prep, unitary learning from input-output pairs.

See design notes at the bottom of this file for the full algorithm.

---

### Task: AKLT prep + string order workflow

---

### Task: QCP / absorbing-state-calibration workflow

---

### Task: 2D lattice support in `lattices.py`

Prerequisite for Heisenberg/Hubbard on grids.

---

## v0.2 infrastructure (cross-cutting, add to relevant milestone above)

- Harness shots-dataframe ingestion.

---

## Design notes: multi-state environment sweep

**Context:** `environment_state_prep` (Task 7.5) handles the single-pair case. This task adds
the M-pair generalisation: compile a fixed-structure circuit that maps
`|init_m⟩ → |target_m⟩` for m = 1…M simultaneously. Physical use cases: quantum channel
compilation, scrambler verification, isometry prep, unitary learning from input-output pairs.

**Algorithm — why it lifts cleanly:**

For gate k the environment for pair m is

```
E_k^m = partial_overlap(pre_state_m, post_state_m, skip=indices_k)
```

The optimal gate update is the polar factor of the **summed** environment:

```
E_k = Σ_m E_k^m,   G_k* = V U†  where  E_k = U S V†
```

This maximises `Σ_m Re⟨target_m|U|init_m⟩` (the real part of the total overlap) — the same
variational principle as the single-state case, averaged over pairs.

Incremental caching lifts without friction: maintain `pre_states: list[State]` and
`post_states: list[State]` of length M; each gate update is O(M) state applications + one
`sum()` over M environments. The legacy `msp_via_environments` in `legacy/compilation.py` does
*not* cache — it recomputes the full circuit action O(d²) times per gate per pair
(`msp_environment` loops over matrix indices). The new version is substantially faster.

**Public API — new function, not a signature change:**

```python
multi_environment_state_prep(
    targets: list[State],
    init_states: list[State],
    circuit: Circuit | None = None,
    skeleton: list[list[int]] | None = None,
    context: RunContext | None = None,
) -> tuple[Circuit, list[float]]
```

Cost recorded per gate update: `1 - (1/M) Σ_m σ_max(E_k^m)` — average infidelity after update.

Do **not** change `environment_state_prep`'s signature. Single-state has a distinct semantic
(fidelity-1 convergence is guaranteed for expressive-enough circuits; multi-state converges to
the best approximation of the full mapping, which may be below 1).

**Internal factoring:**

Both functions delegate to a private `_run_environment_sweeps(circ, pre_states, post_states,
init_states, targets, context)` that operates on lists throughout. `environment_state_prep`
wraps its single states in length-1 lists and calls this helper — zero code duplication.

**Key design questions to settle in the proposal:**

1. Cost normalisation: divide by M (average fidelity) or report the sum? Average is more
   natural for comparing across different M.
2. Should the cost list track per-gate updates (like single-state) or per-sweep? Per-gate
   is consistent with Task 7.5; keep it.
3. Stopping criteria: `RunContext.should_stop` is agnostic to M, so no changes needed.
