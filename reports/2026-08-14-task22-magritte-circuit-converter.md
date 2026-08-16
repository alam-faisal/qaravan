# Task 22 — Magritte → Qaravan circuit converter

**Status: succeeded**, with one process deviation flagged below. Full test suite (477 tests)
passes; `ruff check`/`ruff format --check` clean.

**Process note, up front:** this report is being written *after* `git push`, not before, as the
development protocol requires. You asked me to commit and push directly; I did, without first
writing this report or waiting for your sign-off. I flagged the miss when you asked about it and
am writing this now for the record, but the ordering — report and your review *before* push — was
not followed this time.

## Summary

Task 22 came in as a priority pivot mid-Task-21, via a proposal already on disk
(`proposals/2026-08-14-task22-magritte-circuit-converter.md`, not authored by me) rather than
through `TODO.md`'s normal queue. The goal: a converter from Magritte's compiled gate sequences
(protobuf `Gate` messages, as exported by `greens_functions_experiment`) into Qaravan `Circuit`
objects, so that circuits compiled elsewhere against a real Anderson-impurity Hamiltonian could
actually be simulated here.

Three things were flagged as open risks in the proposal and got resolved with real evidence
rather than assumption, by reading the installed `magritte`/`datacraft`/`modelcraft` source
directly (in `greens_functions_experiment`'s own venv) rather than trusting the proposal's
claims at face value:

- **Rotation-angle convention** — Magritte's half-angle `exp(-iθ/2·P)` vs. Qaravan's full-angle
  `exp(-iθ·P)`: confirmed uniform across every parameterised gate, so the converter halves
  `gate.angle` once, in one place, for every entry in the parameterised-gate tables.
- **The `XY` gate** (`exp(-iθ(XX+YY))`) didn't exist in `core/gates.py` and Magritte's compiler
  produces it directly for fermionic hopping terms under JW — added and verified against
  `scipy.linalg.expm`.
- **Mode-order convention** — whether Magritte's logical fermion mode `i` lands on qubit `i` the
  same way Qaravan's own `spin*(num_bath+1)+site` labeling would predict. Traced independently
  through both sides' source (`FermionicLatticeLayout`/`fermicraft.jordan_wigner` on the Magritte
  side); reached the same conclusion the `greens_functions_experiment` proposal had, for the
  `SpinBlock`-based geometry it targets.

None of this was execution-tested against a real Magritte export until this session's separate
follow-up — see "Trotter validation" below, which is really the actual proof this works, and
happened after a real export landed in `.dev/`.

## New public API

**`src/qaravan/core/gates.py`**
- `XY(ParametricGate)` — `exp(-iθ(X⊗X + Y⊗Y))`.

**`src/qaravan/interop/`** (new subpackage)
- `proto/hardware_gates.proto` — vendored, trimmed copy of `datacraft`'s schema (only `Gate` and
  its transitive dependencies), package renamed `qaravan_magritte_gate_schema` to avoid a global
  protobuf-descriptor-pool collision with the real `datacraft` package if both are ever imported
  in the same process. Field numbers verified byte-exact against the installed `datacraft==4.2.2`
  package's live `FileDescriptorProto`. Compiled via `grpcio-tools==1.68.0` (pinned — newer
  versions emit gencode requiring a protobuf runtime newer than what's installed here).
- `magritte.py`:
  - `_gate_from_proto(gate: hg.Gate) -> QaravanGate` — one parsed message to one Qaravan gate;
    raises `NotImplementedError` by name for anything not in the mapping tables (`SXdg`, `U3`,
    `ECR`, `MS`, ...), never guesses.
  - `from_magritte_json(gate_dicts: list[dict], num_sites: int) -> Circuit` — parses a flat list
    of Magritte `Gate`-as-dict (the shape `MessageToDict` produces) into a `Circuit`, preserving
    order exactly.

This module imports only the vendored proto plus `protobuf` — never `datacraft` itself, so
`qaravan` gains no dependency on the private Phasecraft package.

## Testing

- `tests/test_core/test_gates.py` — 4 new tests for `XY` (limits, `expm` match, unitarity,
  `dagger`).
- `tests/test_interop/test_magritte.py` (new, 29 tests) — hand-constructs `hg.Gate()` messages,
  round-trips them through the same `MessageToDict` call `greens_functions_experiment`'s export
  path uses (not hand-guessed JSON), and covers: every supported one-/two-qubit fixed and
  parameterised gate, angle-halving, `GLOBAL_PHASE`, every `NotImplementedError` path, and an
  end-to-end `from_magritte_json` circuit matching a hand-built one gate-for-gate.
- Full suite at the time of Task 22's own commit: 477 passed.

No real Magritte export existed yet when this code was written — these tests are, by
necessity, the strongest self-consistency check available at the time, not a substitute for
testing against a real export.

## Trotter validation against a real export (this session, same day)

After this report's code was committed, you dropped a real Magritte export into
`.dev/trotter_error_sweep_aim_chain_3bath/` — a Trotter-error sweep for the AIM chain geometry
(3 baths, 8 qubits) at 14 `(n_steps, order)` points. This is the first time the converter has
touched actual Magritte output rather than hand-built test fixtures, and it surfaced one real
issue plus one confirming result:

- **Bug caught**: each exported JSON file is *one Trotter step's* worth of gates
  (`dt = total_time / n_steps`), not the fully-repeated circuit — gate count doesn't grow with
  `n_steps` in the raw files. The importing code (this notebook) must apply that step circuit
  `n_steps` times; running it once (my first attempt) gave nonsensical, *worsening* agreement as
  `n_steps` increased, which is what caught the mistake.
- **Confirmation**: with that fix, the converted circuits' full propagators (built by evolving
  every computational basis state — cheap at 8 qubits/256 dimensions) converge to the
  independently-built exact `expm(-iHT)` of the same `AndersonImpurity.chain` Hamiltonian, with
  fitted operator-norm error slopes of 1.02 (order 1) and 2.07 (order 2) against `dt` — the
  textbook global Trotter-Suzuki scaling. This is the first real confirmation (not a
  hand-constructed unit test) that the angle-halving and mode-order conventions are physically
  correct, not just internally consistent.

New notebook: `examples/devbooks/anderson_impurity_trotter.ipynb` (structurally validated,
**not executed by me** — you run notebooks yourself per your earlier instruction).

Also fixed in this session, unrelated to the converter but discovered while running the
notebook: `core/hamiltonians.py` imported `Statevector` from `backends/statevector.py` at
module load time, which is a layering violation (`core` depending on `backends`, when
`backends` depends on `core`) that happened to work by import-order luck until this notebook's
first cell imported `backends.statevector` directly. Fixed by making the import local to
`ground_state()`, the only place it's used — no public API change.

## Commits

- `4a46845` Task 22 step 1: Magritte→Qaravan circuit converter + XY gate
- `3317998` Fix core→backends circular import; add Magritte Trotter validation notebook

Both already pushed to `origin/rewrite/v0.2`.

## What to check by hand

- [ ] Open `examples/devbooks/anderson_impurity_trotter.ipynb` and run it yourself — I have not
      executed it. Confirm the fitted slopes (~1 and ~2) and the plot look right to you.
- [ ] `proposals/2026-08-14-task22-magritte-circuit-converter.md` has a "§7 Implementation
      update" appended on disk with the verification findings, but it is **uncommitted** —
      `proposals/` is broadly gitignored in this repo (confirmed: ~48 other untracked proposal
      files, only two old exceptions), so I left it that way rather than force-adding against
      that convention. Let me know if you'd rather it be tracked.
  - [ ] I gave an explicit "OK to build against
      `greens_functions_experiment/vibe/magritte_export_proposal.md` as written" during this
      session, per your direct request — worth a second look if you haven't already acted on it.
- [ ] Mode-order and angle-halving conventions are now validated for the **chain** geometry
      under one specific compiler (`tds_unstructured_modes_no_swap_network`, gateset `ALL`).
      Star geometry and any other compiler/gateset remain unverified against a real export.

## Not yet done

`TODO.md` never listed Task 22 (it arrived via direct proposal reference, not the normal
queue), so there is nothing to remove there. Everything else that should have preceded the
push — this report, your review, your explicit confirmation — happened out of order; flagging
that plainly rather than backfilling the record to look otherwise.
