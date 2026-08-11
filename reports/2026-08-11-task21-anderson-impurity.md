# Task 21 — Anderson Impurity Model (star and chain geometries)

**Status: succeeded.** All acceptance criteria from the proposal are met; full test suite
(434 tests) passes; `ruff check`/`ruff format --check` clean on changed files.

## Summary

`AndersonImpurity` follows the exact `FermiHubbard1D` pattern established in Task 20: a bare
`FermionOperator` subclass with no qubits until `.encode(...)` is called. Two named alternate
constructors build the two geometries the source document gives —

- `AndersonImpurity.star(eps_d, U, eps_bath, V)` — impurity coupled directly to every bath mode
  (eq. 1-2 of the source document).
- `AndersonImpurity.chain(eps_d, U, eps_chain, t_chain, V)` — impurity coupled only to the first
  of a nearest-neighbor-hopping nchain (eq. 7-8).

— sharing one type (not two subclasses) since nothing needs to differ at the class level:
`.encode()`'s Trotter-layer re-chunking logic is identical either way, just fed different
`layer_sizes` computed by whichever classmethod built the instance. Per your explicit
confirmation, there is no `num_impurities` parameter — single-orbital only, matching the source
document's own scope.

The star↔chain Lanczos-tridiagonalization mapping (source doc §2.1, eq. 3-7) is **not** library
code, per the approved proposal — but it turned out to matter for testing, not just for users:
without it, star and chain are only comparable at all in the trivial `num_bath=1` case (where
they're forced to be the same system by construction). Mid-build, you asked for the real 5-bath
material parameters to be used in the notebook with a comparison against the equivalent chain —
which required actually implementing that Lanczos derivation, ad-hoc, as a notebook-only numpy
routine (not `core/`), to get real chain parameters to compare against. This produced a much
stronger validation than the unit tests alone can reach: the **full 4096-eigenvalue spectrum**
of the 12-qubit, 5-bath star matches the Lanczos-derived chain's spectrum exactly, not just the
ground energy.

## New public API

**`core/hamiltonians.py`**
- `AndersonImpurity(FermionOperator)` — `num_bath`, `eps_d`, `U`, `topology` (`"star"`/`"chain"`,
  metadata only) attributes; `num_modes = 2*(num_bath+1)`.
  - `.star(eps_d, U, eps_bath, V) -> AndersonImpurity`
  - `.chain(eps_d, U, eps_chain, t_chain, V) -> AndersonImpurity` (`len(t_chain) == len(eps_chain)-1`,
    or `0` if `eps_chain` is empty)
  - `.encode(encoding) -> Hamiltonian` — re-chunks Trotter layers onto the generic
    `FermionOperator.encode()` output, exactly as `FermiHubbard1D.encode()` does.
- `bath_interleaved_mode_order(num_bath) -> tuple[int, ...]` — the qubit layout
  `[b1up, b1down, ..., bNup, bNdown, i_up, i_down]`, as requested.

**Canonical mode convention** (Step A, per `docs/jw_fermi_hubbard_1d.md` §1.1):
`mode(site, spin) = spin*(num_bath+1) + site`, site 0 = impurity, sites 1..num_bath =
bath/chain sites 1..num_bath. This is a code decision (in `_aim_star_terms`/`_aim_chain_terms`),
not forced by physics — "impurity first" could equally have been "impurity last" with no
physical consequence, as discussed.

**Trotter-layer structure** — the one piece of real per-geometry logic:
- Star: **one layer per bath index** (`num_bath + 1` layers total including onsite). Every
  hybridization bond shares the impurity qubit, so unlike a chain there's no even/odd
  parallelization available between different bath indices — each bath's bond must be its own
  sequential Trotter layer. Same-bath-index, opposite-spin bonds *are* paired into one layer
  each (disjoint qubits, free parallelism).
- Chain: identical even/odd bipartition to `_fh_terms`, since impurity+chain sites form one
  ordinary open 1D chain (`LinearLattice(num_bath+1, periodic=False)`), with `V` simply the
  first bond's coupling instead of a uniform `-t`.

Both inherit the Task 20 gate-fusion warning on `trotter_circuit()` automatically (no new code)
since both go through `FermionOperator.encode()`.

**Nothing else needed to be built.** `.encode(...).ground_state(fixed_particle_number=...)`,
`.as_matrix()`, `.as_observable()`, and Majorana observables via `majorana_observable`/
`majorana_bilinear` all work unchanged — confirmed in the notebook (§6, §9).

## Testing

- `tests/test_core/test_anderson_impurity.py` (new, 18 tests): shape/validation for both
  geometries (mismatched-length `ValueError`s, correct `num_modes`); Hermiticity; the atomic
  limit (`num_bath=0`) against the closed-form `{0, ε_d, ε_d, 2ε_d+U}` spectrum for both
  geometries; star≡chain at `num_bath=1`; particle-number conservation via
  `fixed_particle_number`; `trotter_circuit()` vs exact propagator (both geometries, confirming
  the inherited gate-fusion warning fires); `bath_interleaved_mode_order` correctness
  (hand-computed for `num_bath=1,2`), permutation validity, and encoding-invariance of the
  spectrum under it.
- One `LinearLattice` edge case surfaced and was handled: `LinearLattice(n, ...)` requires
  `n >= 2`, so `_aim_chain_terms` guards the `num_bath == 0` atomic-limit case (no chain sites
  at all) to skip the `LinearLattice` call entirely rather than constructing one of size 1.
  (This same latent edge case exists in `_fh_terms` for `FermiHubbard1D(1)` — not fixed here,
  out of scope, noted for awareness.)
- Full suite: **434 passed**, 0 failed.

## Commits

- `6df4cb5` AndersonImpurity (star and chain geometries)
- `93bd012` notebook: AndersonImpurity star/chain, real 5-bath material example

## What to check by hand

- [ ] Open `examples/devbooks/anderson_impurity.ipynb`. §5 is the one worth reading closely —
      it derives chain parameters from the real 5-bath star parameters via Lanczos
      tridiagonalization and confirms the full spectra agree exactly. §9 computes a Majorana
      expectation value on the impurity (motivated by the source document's Green's function
      definition, eq. 12) — not the Green's function itself, just the observable.
- [ ] Confirm you're comfortable with the Lanczos-derivation code living only in the notebook
      (not `core/`) despite now clearly working correctly — it was written strictly for
      generating a test case, per the proposal's scope boundary, not as a reusable conversion
      utility.
- [ ] The source document's Table 1 values are used directly in the notebook, as instructed —
      confirm you're comfortable with them being in a committed notebook file (not just `src/`),
      given the document's confidentiality marking. I did not include them anywhere else
      (proposal, this report, or any `src/` file).
- [ ] Sanity-check `bath_interleaved_mode_order`'s formula against your own mental model from
      the design discussion — it's a direct transcription of that conversation, worth a second
      look given it's easy to get an off-by-one wrong in this kind of permutation code.

## Not yet done (deliberately, per protocol)

TODO.md still lists Task 21 (not removed), and nothing has been pushed. Both wait on your
review and confirmation, per the development protocol. Green's function computation (source
doc §3, §3.1) remains explicitly out of scope — its own future task once you're ready for it.
