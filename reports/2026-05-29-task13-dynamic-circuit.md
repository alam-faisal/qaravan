# Task 13 Report: DynamicCircuit abstraction

**Status:** Success

---

## What was accomplished

### New abstractions in `core/dynamic_circuit.py`

```python
@dataclass
class DynamicRound:
    meas_sites: list[int]
    decoder: Callable[[str], Circuit]

@dataclass
class DynamicCircuit:
    rounds: list[DynamicRound]

    def run(self, init_state: State) -> tuple[State, list[str]]: ...
```

- `decoder` is always called: `""` for rounds with `meas_sites=[]` (constant-function idiom: `lambda _: circuit`)
- `run` always returns `(State, list[str])` — outcomes has one entry per measured round
- Backend-agnostic: `run` calls `state.apply()` which dispatches to the default simulator

### New application library `applications/dynamic_circuit_library.py`

```python
def ghz_fusion_dqc(n, cluster_size=3, use_cancellations=False) -> DynamicCircuit
def build_ghz_decoder(cluster_size, num_clusters, total_qubits, use_cancellations=False) -> Callable[[str], Circuit]
def _kept_sites(cluster_idx, cluster_size, num_clusters) -> list[int]
```

`ghz_fusion_dqc` builds a 2-round DQC:
- Round 0: `meas_sites=[]`, `decoder = lambda _: prep_circuit + bell_circuit`
- Round 1: `meas_sites=boundary_qubits`, `decoder = build_ghz_decoder(...)`

### Code removed

`ghz_via_fusion`, `build_ghz_decoder`, `_kept_sites` removed from `compilation.py`. `ghz_via_fusion` removed from the codebase entirely — the notebook now uses `ghz_fusion_dqc` + `dqc.run()` directly.

---

## Tests

**33 new tests** across two files; total suite 318 passing.

### `tests/test_core/test_dynamic_circuit.py` (8 tests)

| Test | What it catches |
|---|---|
| `no_measurement_passes_empty_string` | Decoder not called or called with wrong arg on unmeasured round |
| `measurement_calls_decoder_with_outcome` | Decoder called with `""` instead of real outcome |
| `run_applies_rounds_in_order` | Reversed or skipped rounds |
| `does_not_mutate_init_state` | Shared-reference bug on input tensor |
| `returns_only_measurement_outcomes` | outcomes list including entries for no-measurement rounds |
| `outcome_string_lengths` | Off-by-one in `meas_sites` |
| `empty_correction_is_no_op` | Crash or state corruption on empty correction circuit |
| `two_measurement_rounds` | outcomes list not updated on second measured round |

### `tests/test_applications/test_dynamic_circuit_library.py` (12 tests)

| Test | What it catches |
|---|---|
| `kept_sites_*` (3 tests) | Wrong boundary qubit formula for first/last/interior clusters |
| `decoder_empty_outcome_gives_no_correction` | Spurious correction for `"00"` |
| `decoder_cancellations_agree_with_naive` | Cancellation folding: wrong sign or wrong qubit (all 4 n=4 outcomes) |
| `dqc_raises_*` (2 tests) | Bad input validation |
| `dqc_has_two_rounds`, `round0_no_measurement`, `round1_meas_sites` | DQC structure |
| `dqc_4qubit_fidelity` | Wrong correction for any of the 4 outcomes (20 runs) |
| `dqc_4qubit_all_outcomes_covered` | Correction logic silently failing for a specific outcome (100 runs) |
| `dqc_6qubit_fidelity` | Decoder error for multi-boundary case (20 runs) |
| `dqc_use_cancellations_same_physics` | Cancellation path disagreeing with naive path on actual GHZ output |

---

## Things to verify by hand

1. Open `examples/devbooks/ghz_fusion.ipynb` and re-run cell 7 several times — each run should print a different outcome (`00`, `01`, `10`, `11`), a matching correction circuit, and fidelity = 1.0.
2. Run the statistical verification cell (cell 9) — should confirm all 4 outcomes appear and all 200 fidelities = 1.0.
3. Run the multi-boundary cell (cell 11) — n=6, 12 physical qubits, 6-bit outcome, fidelity = 1.0 for 20 runs.
4. Confirm that `ghz_via_fusion` is no longer importable from `qaravan.applications`.
