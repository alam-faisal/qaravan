# TODO.md — Qaravan v0.2 rewrite

This file is the active task list for the v0.2 rewrite. Entries are roughly
ordered; each has explicit acceptance criteria. Treat the criteria as the
definition of done.

Before implementing a task review the "Development protocol" and "Coding style" sections in CLAUDE.md.

---

## v0.2.2 — Track B: Hamiltonians and Fermions (current priority)

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

## Task 15 — Heisenberg 1D in `core/hamiltonians.py`

**Goal:** Add `Heisenberg1D` to `hamiltonians.py`. Minimal but extensible.

`H = J Σ_i (XX_i + YY_i + ZZ_i) + h Σ_i Z_i`

Trotter step uses existing RXX, RYY, RZZ, RZ gates. Keep everything 1D for now; generalisation
to 2D and other lattice geometries comes when `lattices.py` is developed (v0.2.3+).

**Interface (mirror TFI exactly):**
- `Heisenberg1D(n, J=1.0, h=0.0)` — constructor
- `.trotter_circuit(dt, order=2) -> Circuit`
- `.as_observable() -> Observable` (PauliString sum)
- `.ground_state(method='exact') -> Statevector` — via `numpy.linalg.eigh` on the full matrix

**Tests:**
- `test_heisenberg_ground_state_energy_j0`: at J=0, ground state of +h Z-field is all-|0⟩,
  energy = -h*n.
- `test_heisenberg_ground_state_against_exact_diag`: n ≤ 6, compare to numpy.linalg.eigh.
- `test_heisenberg_trotter_vs_propagator`: 2nd-order Trotter at small dt vs `expm(-iHt)`.
- `test_heisenberg_xx_model_degeneracy`: at h=0, ground state is degenerate (singlet) for
  even n — verify ground energy matches known result.

---

## Task 16 — Fermi-Hubbard 1D with Jordan-Wigner mapping

**Goal:** Add `FermiHubbard1D` to `hamiltonians.py`. The primary deliverable is a clean
Jordan-Wigner encoding of the Hamiltonian as a qubit operator, enabling exact diagonalization
and Trotter evolution. DMRG via MPS comes later; this task needs only the Statevector backend.

**Hamiltonian:**
`H = -t Σ_{i,σ} (c†_{i,σ} c_{i+1,σ} + h.c.) + U Σ_i n_{i↑} n_{i↓} + μ Σ_{i,σ} n_{i,σ}`

with σ ∈ {↑, ↓}, and the standard Jordan-Wigner strings.

**Encoding:** 2n qubits. Spin-up occupancies on qubits 0..n-1, spin-down on qubits n..2n-1.
This ordering means the JW string for a spin-down operator threads through all spin-up qubits:
`c†_{j,↓} = (Π_{i=0}^{n-1} Z_i)(Π_{i=n}^{n+j-1} Z_i)(X_{n+j} - iY_{n+j})/2`.
The spin-up JW string is local: `c†_{j,↑} = (Π_{i=0}^{j-1} Z_i)(X_j - iY_j)/2`.

After JW the Hamiltonian becomes a sum of Pauli strings (all weight ≤ 2n). This Pauli sum
is the output of `FermiHubbard1D.as_pauli_sum()` and is used for:
- Exact diagonalization: build the full matrix via `as_pauli_sum().as_matrix(n_qubits=2n)`,
  call `numpy.linalg.eigh`, return ground state as `Statevector`.
- Trotter evolution: each Pauli string exponentiates to a product of single-site rotations and
  RXX/RYY/RZZ gates.

**Interface:**
- `FermiHubbard1D(n_sites, t=1.0, U=1.0, mu=0.0)` — n_sites is the number of lattice sites,
  total qubits = 2*n_sites.
- `.as_pauli_sum() -> PauliSum`
- `.as_matrix() -> np.ndarray` — full 4^n Hamiltonian matrix (for small n)
- `.ground_state(n_electrons: int | None = None) -> Statevector` — via exact diag; optionally
  restrict to a fixed electron number sector
- `.trotter_circuit(dt, order=1) -> Circuit`

**Tests:**
- `test_fh_jw_hopping_term_is_hermitian`: each hopping Pauli string pair is self-adjoint.
- `test_fh_ground_state_n1_analytical`: n=1 site, 2 qubits; ground state energy is
  `(U/2) - sqrt((U/2)² + t²)` for half-filling — compare to exact diag.
- `test_fh_ground_state_agrees_with_known_2site`: n=2 at half-filling (2 electrons); compare
  to published Hubbard dimer spectrum.
- `test_fh_as_matrix_is_hermitian`: `np.allclose(H, H.conj().T)`.
- `test_fh_electron_number_conservation`: ground state in fixed-N sector has the right
  particle number (verify via `n_up + n_down` expectation value).
- `test_fh_trotter_vs_propagator`: small dt, small n.

**Notes:**
- Implement JW string construction as a private helper `_jw_op(site, spin, n_sites)` that
  returns the Pauli string. This is the heart of the task — get it right and everything else
  follows.
- No lattice.py needed; everything stays 1D for now.
- The `PauliSum` backend (Task 12+ in the original list) may not be implemented yet. If so,
  `as_pauli_sum()` can return a `list[tuple[complex, PauliString]]` as an interim format and
  be upgraded when `PauliSum` lands.

---

## v0.2.2 — Track A: Dynamic Quantum Circuits (deferred until Track B complete)

---

## Task 14 — Environment sweep for DQCs

**Goal:** Extend `environment_state_prep` to handle `DynamicCircuit` targets: optimize the
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

**Do NOT start this task without the Julia code as reference** — the normalization subtleties
are easy to get wrong. The Julia code is in `legacy/julia/train_dqc.jl` (or wherever it ends up).

**Acceptance:** Notebook shows `dqc_environment_state_prep` learning the correct decoder
circuit for GHZ-from-clusters for a 4-qubit target.

---

## v0.2 infrastructure spine (after v0.2.2 tracks are complete)

---

## Task 8 — Concrete `NoiseModel` subclasses in `core/noise.py`

**Goal:** Port `ThermalNoise`, `PauliNoise`, `PauliLindbladNoise` to the new ABC.

**Tests:**
- `test_thermal_noise_superoperator_cptp`: the superoperator is CPTP.
- `test_pauli_noise_as_pauli_channel`: sums to probability 1.
- `test_thermal_noise_as_pauli_channel_raises`: clear error message.

---

## Task 9 — Implement second backend: `DensityMatrix` and `DensityMatrixSimulator`

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

## Task 11 — First end-to-end workflow: noisy Trotter sweep

**Goal:** The workflow in CLAUDE.md "Noisy Trotter evolution with an observable" runs end-to-end.
Acceptance test for phases 0-2 of the v0.2 rewrite.

**Acceptance:**
- The code block in CLAUDE.md runs exactly as written.
- Output magnetization curve matches v0.1 output (legacy reference) within numerical tolerance.

---

## v0.2.3 backlog

The following are queued but explicitly deferred until the v0.2.2 tracks are complete:

- MPS backend + MPS-vs-SV cross-checks.
- MPDO backend + MPDO-vs-DM cross-checks.
- Monte Carlo backend with `Walkers`.
- Clifford backend (wraps Stim).
- Matchgate backend (with non-Gaussian extension on the roadmap).
- Pauli propagation backend (wraps `PauliPropagation.jl`).
- Multi-state environment sweep (`multi_environment_state_prep`). See design notes below.
- AKLT prep + string order workflow.
- QCP / absorbing-state-calibration workflow.
- Harness shots-dataframe ingestion.
- 2D lattice support in `lattices.py` (prerequisite for Heisenberg/Hubbard on grids).

Each will get its own proposal and acceptance criteria when promoted to active.

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

---
