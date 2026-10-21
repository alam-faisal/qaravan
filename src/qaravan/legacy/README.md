# Legacy — Frozen v0.1 code

This directory contains the original v0.1 source, flattened from its former
`core/`, `tensorQ/`, `algebraQ/`, and `applications/` subdirectory structure.
It is kept as read-only reference material during the v0.2 rewrite.

**Do not add features or fix bugs here.** Do not copy patterns from this code
into v0.2 without checking the known issues below first.

---

## Known issues in v0.1 (do not replicate in v0.2)

1. **`StatevectorSim` silently ignores noise channels.** If you pass `nm=...`
   to `StatevectorSim`, the noise is dropped without warning. v0.2 raises
   `IncompatibleNoiseError` instead.

2. **`DensityMatrixSim.local_expectation` mishandles multi-site operators.**
   It contracts each site independently. Only correct for single-site
   observables. v0.2 unifies under `state.expectation(obs)` which handles
   multi-site correctly.

3. **API drift across backends.** `StatevectorSim` has `pauli_expectation`,
   `local_expectation`, `one_local_expectation`. `DensityMatrixSim` has
   `local_expectation` and `global_expectation`. `MPSSim` has
   `one_local_expectation` only. v0.2 unifies under `state.expectation(obs)`.

4. **`MPSSim.local_expectation` returns `None`.** Use
   `mps.one_local_expectation(site, op)` instead, or extract via
   `mps.to_vector()` for small systems.

5. **`CNOT` defaults to `n=1000`.** Magic default exists due to PBC boundary
   detection logic. Pass `n=` explicitly if PBC matters.
