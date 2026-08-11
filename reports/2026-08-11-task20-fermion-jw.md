# Task 20 — Generic fermion-to-qubit encoding: `FermionOperator` + Jordan-Wigner transform

**Status: succeeded.** All acceptance criteria from the proposal are met; full test suite
(416 tests) passes; `ruff check`/`ruff format --check` clean on changed files.

## Summary

Fermionic models are now defined once, encoding-agnostic, and only become qubits when
explicitly asked via `.encode(...)`. `FermiHubbard1D` no longer takes an `ordering=` argument
or produces a qubit-space object at construction; the encoding (Jordan-Wigner, and which qubit
each mode lands on) is a first-class value (`FermionicEncoding`) chosen once and reused for
both the Hamiltonian and any observable evaluated against its ground state — closing a silent
wrong-answer bug class where the two could otherwise independently drift onto different
encodings.

The design changed twice during implementation from what was in the approved proposal, both
discovered by direct testing rather than by re-reading the proposal:

1. **`FermionicHamiltonian` was eliminated.** The approved proposal had it as a separate class
   requiring `trotter_groups` at construction — exactly the "Trotter is a construction-time
   decision" problem flagged mid-review. It's gone; `FermionOperator` itself (already flat,
   already existed) grew the `.encode()` method, and `FermiHubbard1D` subclasses it directly,
   attaching its own Trotter-layer bookkeeping only *after* calling the generic, Trotter-
   agnostic `.encode()`.
2. **`jordan_wigner`'s signature changed from grouped to flat, plus automatic Hermiticity
   pairing.** The approved proposal's `jordan_wigner(term_groups: list[list[FermionTerm]], ...)`
   took caller-specified fusion groups. Flattening it to `jordan_wigner(terms: list[FermionTerm], ...)`
   surfaced a real correctness gap: `Hamiltonian`'s existing, shipped `_trotter_order1/2`
   exponentiate each `HamiltonianTerm` individually, which is only unitary if that term is
   Hermitian — and a bare `c†_a c_b` isn't. The fix (not in the original proposal, added during
   build): `jordan_wigner` JW-maps each term and auto-detects whether the result already has
   real coefficients; if not, it searches the remaining terms for whichever one's image sums to
   something real (exactly what `hopping()`'s two halves do for each other), and raises
   `ValueError` if no such partner exists. This is derived from the algebra, not caller-specified
   — it doesn't reintroduce the grouping problem.

Two smaller items were added mid-build at your request, beyond the original proposal:

- **`Hamiltonian.ground_state(fixed_particle_number=...)`** — generalizes what was previously
  `FermiHubbard1D.ground_state(n_electrons=...)`. Now works on *any* qubit `Hamiltonian`
  (resolves proposal open question 4 in favor of the general option).
- **Gate-fusion warning on `trotter_circuit()`.** `Hamiltonian` gained a `_gate_fusion_applied`
  flag (default `True`, so `TFI`/`Heisenberg1D` are unaffected). `FermionOperator.encode()` sets
  it `False`; `trotter_circuit()` warns (once, doesn't block) when building from such a
  Hamiltonian, since per-term flattening splits FH's old fused onsite gate into 3. This is the
  partial, in-scope fix for the broader "Hamiltonian methods presuppose a solver/circuit
  strategy" critique — a full fix (free functions instead of methods) would touch shipped
  `TFI`/`Heisenberg1D` code and is out of scope here.

## New/changed public API

**`core/observables.py`**
- `PauliString.compose(other) -> PauliString | PauliSum`, `__matmul__` — operator product
  (matrix multiplication), not tensor product.
- `PauliSum.compose(other) -> PauliSum`, `__matmul__`.
- `PauliSum.simplify(atol=1e-12) -> PauliSum` — merges duplicate strings, drops near-zero terms.

**`core/fermions.py`** (new module)
- `FermionTerm(coeff, ops)`, `FermionOperator(terms, num_modes)` — flat ladder-operator algebra;
  `.dagger()`, `.compose()`/`@`, `*`, `+`; `FermionOperator.encode(encoding) -> Hamiltonian`.
- `c(mode)`, `cdag(mode)`, `number(mode)`, `hopping(a, b, coeff) -> list[FermionTerm]`,
  `density_density(a, b, coeff) -> list[FermionTerm]`.
- `jordan_wigner(terms, num_modes, mode_order=None) -> list[HamiltonianTerm]` — flat in, flat
  out, automatic Hermitian-pairing (see above); raises `ValueError` on an unpairable term.
- `MajoranaTerm(coeff, ops)`, `MajoranaOperator(terms, num_modes)`, `gamma(index)`,
  `majorana_bilinear(a, b, coeff=1j)`, `to_majorana(fermion_term) -> MajoranaOperator`.
- `majorana_observable(terms, num_modes, mode_order=None) -> PauliSum` — full-register
  observable, ready for `state.expectation(...)`.
- `FermionicEncoding(name, num_modes, mode_order=None)` — frozen, hashable;
  `.encode_fermion_terms(...)`, `.encode_majorana_terms(...)`, `.qubit_of(mode)`.

**`core/hamiltonians.py`**
- `Hamiltonian.ground_state(method="exact", fixed_particle_number=None)` — new
  `fixed_particle_number` param (see above).
- `Hamiltonian.trotter_circuit(...)` — now warns (`UserWarning`, once per call) when
  `_gate_fusion_applied` is `False`.
- `FermiHubbard1D(num_fermions, t=1.0, U=1.0, mu=0.0, periodic=False)` — **no `ordering=`
  argument**; subclasses `FermionOperator`; `.encode(encoding) -> Hamiltonian` override attaches
  its own Trotter-layer structure.
- `interleaved_mode_order(num_fermions) -> tuple[int, ...]` — the interleaved-layout permutation;
  "stacked" is just `FermionicEncoding`'s default (no equivalent helper needed).

## Testing

- `tests/test_core/test_observables.py`: +11 tests for `compose`/`simplify` (single-qubit
  identities, associativity on random triples, matches direct matrix products, distributes
  over `PauliSum`, the raising/lowering→number-operator identity, merge/drop behavior).
- `tests/test_core/test_fermions.py` (new, 21 tests): ladder-operator JW mapping (number,
  weight-2 and weight-3 hopping forms cross-checked against `docs/jw_fermi_hubbard_1d.md`),
  `mode_order` respected, zero-terms dropped, unpaired-term `ValueError`, `FermionOperator`
  algebra and `.encode()`, `FermionicEncoding` equality/hashing, Majorana algebra and
  `majorana_observable` (including a from-scratch hand-derivation cross-check that caught a
  sign error in my own test comment, not the implementation — see commit `42b94df`).
- `tests/test_core/test_fermi_hubbard.py`: all pre-existing assertions preserved, call sites
  updated to `fh.encode(encoding)`; added `.encode(stacked)`/`.encode(interleaved)` spectrum
  agreement and a `pytest.warns` test for the gate-fusion warning.
- Full suite: **416 passed**, 0 failed. Pre-existing `RuntimeWarning`s from `circuit.py`
  (`divide by zero`/`overflow encountered in matmul`) during Trotter-circuit matrix
  construction were confirmed unrelated to this task — they reproduce identically for
  unmodified `TFI`, so not investigated further here.

## Commits

- `0659d39` Pauli operator composition
- `42b94df` FermionOperator/MajoranaOperator algebra + Jordan-Wigner transform
- `a96a95c` FermiHubbard1D becomes a bare FermionOperator + encode()

## What to check by hand

- [ ] Open `examples/devbooks/fermions_jw.ipynb` and read through — it follows exactly the
      9-step outline you gave, plus a `FermiHubbard1D`-specific section and the Trotter warning
      demo. All cells executed cleanly (`jupyter nbconvert --execute`); step 9's dimer amplitudes
      and the stacked/interleaved energy agreement are worth a manual glance.
- [ ] Sanity-check the Hermiticity-pairing logic in `jordan_wigner` (`core/fermions.py`) — it's
      the one piece of real new logic (not just plumbing) added beyond the approved proposal.
- [ ] Confirm you're happy with `_gate_fusion_applied` living on `Hamiltonian` as a leading-
      underscore constructor kwarg, versus some other mechanism.
- [ ] Review `docs/jw_fermi_hubbard_1d.md` cross-references in `_fh_terms`/tests — I didn't
      change that document; it still describes the physics, just not literally the code shape
      anymore (no more hand-typed Pauli strings).
- [ ] Note: `to_majorana` and `PauliSum.simplify()` were flagged as open questions in the
      proposal and I implemented both (simplify because `jordan_wigner` needs it internally
      anyway; `to_majorana` because it was cheap and you'd asked about Majorana support) — flag
      if you'd rather either be trimmed.

## Not yet done (deliberately, per protocol)

TODO.md still lists Task 20 (not removed), and nothing has been pushed. Per the development
protocol, both happen only after you've reviewed this report and confirmed. Task 21 (Anderson
impurity) is next and unstarted — I'll write its proposal once you're satisfied here.
