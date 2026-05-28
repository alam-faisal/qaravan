# Proposal: Task 13 — `DynamicCircuit` abstraction

## Overview

`DynamicCircuit` is a container for multi-round DQC protocols. It grows directly out of the GHZ cluster-fusion idiom from Task 12 and is designed to be the substrate for the environment-sweep optimizer in Task 14.

---

## Design

### `DynamicRound`

```python
@dataclass
class DynamicRound:
    meas_sites: list[int]              # sites to measure; [] = no measurement this round
    decoder: Callable[[str], Circuit]  # outcome_str -> circuit to apply after measurement
```

The decoder is always called, even when `meas_sites=[]`. For no-measurement rounds, it receives `""` and should return the circuit unconditionally (a constant function: `lambda _: some_circuit`). Rounds with `meas_sites=[]` never call `measure_and_collapse`, so there is no empty-list edge case to handle in the backend.

### `DynamicCircuit`

```python
@dataclass
class DynamicCircuit:
    rounds: list[DynamicRound]

    def run(self, init_state: State) -> tuple[State, list[str]]:
        sv = init_state
        outcomes: list[str] = []
        for rnd in self.rounds:
            if rnd.meas_sites:
                sv, outcome = sv.measure_and_collapse(rnd.meas_sites)
                outcomes.append(outcome)
            else:
                outcome = ""
            circuit = rnd.decoder(outcome)
            if circuit.gates:
                sv = sv.apply(circuit)
        return sv, outcomes
```

`outcomes` has one entry per round with non-empty `meas_sites`. Always returns a tuple — no `verbose` flag.

### Constant decoder idiom

For no-measurement rounds the decoder ignores its argument and always returns the same circuit. Python idiom: `lambda _: circuit`. The `_` signals the argument is intentionally unused. No helper function needed.

---

## File layout

### New files

- **`src/qaravan/core/dynamic_circuit.py`** — `DynamicRound` and `DynamicCircuit` only. In `core/` to signal it is a first-class abstraction alongside `Circuit`.
- **`src/qaravan/applications/dynamic_circuit_library.py`** — GHZ-specific builders. Parallel to `circuit_library.py`.

### Changes to existing files

**`src/qaravan/core/__init__.py`** — add `DynamicRound`, `DynamicCircuit` to imports and `__all__`.

**`src/qaravan/applications/compilation.py`** — remove `ghz_via_fusion`, `build_ghz_decoder`, `_kept_sites`. The `environment_state_prep` and its private helpers stay.

**`src/qaravan/applications/__init__.py`** — remove `ghz_via_fusion`, `build_ghz_decoder`; add `ghz_fusion_dqc`.

**`tests/test_applications/test_compilation.py`** — remove `ghz_via_fusion` tests (they move to `test_dynamic_circuit_library.py`). The `environment_state_prep` tests are unaffected.

---

## New code

### `core/dynamic_circuit.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from qaravan.core.base import State
from qaravan.core.circuits import Circuit


@dataclass
class DynamicRound:
    meas_sites: list[int]
    decoder: Callable[[str], Circuit]


@dataclass
class DynamicCircuit:
    rounds: list[DynamicRound]

    def run(self, init_state: State) -> tuple[State, list[str]]:
        sv = init_state
        outcomes: list[str] = []
        for rnd in self.rounds:
            if rnd.meas_sites:
                sv, outcome = sv.measure_and_collapse(rnd.meas_sites)
                outcomes.append(outcome)
            else:
                outcome = ""
            circuit = rnd.decoder(outcome)
            if circuit.gates:
                sv = sv.apply(circuit)
        return sv, outcomes
```

### `applications/dynamic_circuit_library.py`

```python
def _kept_sites(cluster_idx: int, cluster_size: int, num_clusters: int) -> list[int]:
    ...  # moved from compilation.py unchanged

def build_ghz_decoder(
    cluster_size: int,
    num_clusters: int,
    total_qubits: int,
    use_cancellations: bool = False,
) -> Callable[[str], Circuit]:
    ...  # moved from compilation.py unchanged

def ghz_fusion_dqc(
    n: int,
    cluster_size: int = 3,
    use_cancellations: bool = False,
) -> DynamicCircuit:
    """DynamicCircuit for n-qubit GHZ via cluster fusion.

    Round 0: constant decoder applies prep_circuit + bell_circuit (no measurement).
    Round 1: measures boundary qubits, decoder returns Pauli correction.
    """
    if cluster_size <= 2:
        raise ValueError(f"cluster_size must be > 2; got {cluster_size}")
    if (n - 2) % (cluster_size - 2) != 0:
        raise ValueError(
            f"(n-2)={(n - 2)} must be divisible by (cluster_size-2)={(cluster_size - 2)}"
        )
    num_clusters = (n - 2) // (cluster_size - 2)
    total_qubits = num_clusters * cluster_size

    prep_circuit = Circuit([], num_sites=total_qubits)
    for cluster_idx in range(num_clusters):
        cluster_sites = list(range(cluster_idx * cluster_size, (cluster_idx + 1) * cluster_size))
        prep_circuit = prep_circuit + ghz_cluster_prep_circuit(cluster_sites, total_qubits)

    boundary_qubits: list[int] = []
    bell_circuit = Circuit([], num_sites=total_qubits)
    for boundary_idx in range(num_clusters - 1):
        left_qubit = (boundary_idx + 1) * cluster_size - 1
        right_qubit = (boundary_idx + 1) * cluster_size
        boundary_qubits.extend([left_qubit, right_qubit])
        bell_circuit = bell_circuit + bell_basis_circuit(left_qubit, right_qubit, total_qubits)

    prep_and_bell = prep_circuit + bell_circuit
    round_0 = DynamicRound(
        meas_sites=[],
        decoder=lambda _: prep_and_bell,
    )
    round_1 = DynamicRound(
        meas_sites=boundary_qubits,
        decoder=build_ghz_decoder(cluster_size, num_clusters, total_qubits, use_cancellations),
    )
    return DynamicCircuit(rounds=[round_0, round_1])
```

---

## Notebook changes (`ghz_fusion.ipynb`)

The step-by-step cell becomes:

```python
from qaravan.core.dynamic_circuit import DynamicCircuit
from qaravan.applications import ghz_fusion_dqc

n, cluster_size = 4, 3
dqc = ghz_fusion_dqc(n, cluster_size)
init = Statevector(bitstring="0" * dqc.rounds[0].decoder("").num_sites)
sv, outcomes = dqc.run(init)
# ... inspect sv, outcomes, fidelity ...
```

The statistical verification cells (currently calling `ghz_via_fusion`) are rewritten as the 3-line equivalent: `dqc = ghz_fusion_dqc(n, cs); sv, outcomes = dqc.run(init); <fidelity check>`.

---

## Tests

### `tests/test_core/test_dynamic_circuit.py`

```python
def test_dynamic_round_no_measurement_passes_empty_string():
    """meas_sites=[] passes '' to decoder; decoder circuit is applied."""

def test_dynamic_round_measurement_calls_decoder_with_outcome():
    """meas_sites non-empty: decoder receives the actual outcome string."""

def test_dynamic_circuit_run_applies_rounds_in_order():
    """Two-round circuit: state after run() matches manual application."""

def test_dynamic_circuit_run_does_not_mutate_init_state():
    """init_state is unchanged after run()."""

def test_dynamic_circuit_run_returns_only_measurement_outcomes():
    """outcomes list length == number of rounds with non-empty meas_sites."""

def test_dynamic_circuit_outcome_string_lengths():
    """Each outcome string length == len(meas_sites) for that round."""

def test_dynamic_circuit_empty_correction_not_applied():
    """Decoder returning Circuit([], num_sites=N) does not crash and leaves state unchanged."""
```

### `tests/test_applications/test_dynamic_circuit_library.py`

```python
def test_ghz_fusion_dqc_raises_for_small_cluster_size():
    """cluster_size <= 2 raises ValueError."""

def test_ghz_fusion_dqc_raises_for_non_integer_num_clusters():
    """(n-2) % (cluster_size-2) != 0 raises ValueError."""

def test_ghz_fusion_dqc_structure():
    """2 rounds; round_0 has meas_sites=[]; round_1 meas_sites == boundary_qubits."""

def test_ghz_fusion_dqc_4qubit_fidelity():
    """20 runs, fidelity with |GHZ_4> on kept sites = 1.0."""

def test_ghz_fusion_dqc_4qubit_all_outcomes_covered():
    """100 runs cover all 4 outcome strings."""

def test_ghz_fusion_dqc_6qubit_fidelity():
    """n=6, cluster_size=3: 20 runs, fidelity = 1.0, outcome length = 6."""

def test_build_ghz_decoder_use_cancellations():
    """use_cancellations=True produces identical final state to use_cancellations=False."""
```

### What these tests catch vs. don't catch

| Test | Catches | Misses |
|---|---|---|
| `no_measurement_passes_empty_string` | Decoder not called / called with wrong arg for empty round | Decoder logic errors |
| `run_applies_rounds_in_order` | Reversed or skipped round | Wrong gate within a round |
| `does_not_mutate_init_state` | Shared-reference bug | Circuit mutation |
| `returns_only_measurement_outcomes` | outcomes list includes empty-round entries | Wrong outcome content |
| `outcome_string_lengths` | Off-by-one in meas_sites | Outcome content |
| `dqc_4qubit_fidelity` | Wrong correction for any outcome | Bugs common to both correction paths |
| `dqc_all_outcomes_covered` | Silent failure for specific outcomes | Bugs in specific n/cluster_size combos |
| `use_cancellations` | Cancellation logic disagrees with naive logic | Both paths wrong the same way |
