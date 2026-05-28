# Report: Task 12 — GHZ state prep via cluster fusion

## Status: SUCCESS

---

## What was accomplished

Three new functions + one infrastructure bugfix.

### New functions

**`applications/circuit_library.py`**

```python
def bell_basis_circuit(a: int, b: int, num_sites: int) -> Circuit:
    """CNOT(a→b) then H(a); pre-rotation before Bell-basis measurement of qubits a, b."""
```

```python
def ghz_cluster_prep_circuit(cluster_sites: list[int], num_sites: int) -> Circuit:
    """GHZ state on cluster_sites in a num_sites-qubit register.
    H(cluster_sites[0]), CNOT(cluster_sites[0], s) for s in cluster_sites[1:].
    """
```

**`applications/compilation.py`**

```python
def ghz_via_fusion(n: int, k: int = 3) -> tuple[Statevector, list[str]]:
    """Prepare n-qubit GHZ via mid-circuit-measurement fusion of k-qubit clusters.
    Returns (final_sv, outcomes) where final_sv spans C*k physical qubits.
    Requires k > 2 and (n - 2) % (k - 2) == 0.
    """
```

Private helper (no tests required):
```python
def _cluster_kept_sites(j: int, k: int, C: int) -> list[int]
```

All three public functions exported from `applications/__init__.py`.

### Infrastructure bugfix: `construct_layers()` in `core/circuits.py`

The greedy layer-packing algorithm had a silent bug: it could place a gate into an
earlier layer than its sequential predecessors allowed. Example: in H(0), CNOT(0→1),
CNOT(1→2), the algorithm placed CNOT([1,2]) in layer 0 alongside H([0]) (no site
conflict), but CNOT([1,2]) must follow CNOT([0,1]) which is in layer 1.

Fix: track `layer_of_site[s]` = latest layer index for site `s`; each new gate starts
from `max(layer_of_site[s]+1 for s in gate.indices)` instead of layer 0.

The bug only manifests in chain-topology circuits (GHZ via chain CNOT). Star-topology
circuits (brickwall, `ghz_cluster_prep_circuit`) are unaffected, which is why it was
previously hidden. The fix is backwards-compatible: all existing tests still pass.

### `docs/API_VISION.md`

Fixed the `measure_and_collapse` return-order example to match the implementation:
`sv, outcome = sv.measure_and_collapse(meas_sites)` (was `outcome, sv = ...`).

---

## Tests added (13 new, 287 total)

### `test_circuit_library.py`

| Test | Catches | Does NOT catch |
|------|---------|----------------|
| `test_bell_basis_circuit_maps_bell_states_to_comp_basis` | Wrong CNOT/H order, wrong site | Multi-qubit labeling errors (2-qubit test) |
| `test_bell_basis_circuit_num_sites` | Wrong num_sites in circuit | — |
| `test_ghz_cluster_prep_circuit_contiguous_gives_ghz_state` | Implementation bugs in cluster prep | Chain-vs-star topology differences |
| `test_ghz_cluster_prep_circuit_offset_sites` | Site-offset errors | — |

### `test_compilation.py`

| Test | Catches |
|------|---------|
| `test_ghz_via_fusion_raises_for_k_le_2` | Missing k≤2 validation |
| `test_ghz_via_fusion_raises_for_non_integer_C` | Missing (n-2)%(k-2)≠0 validation |
| `test_ghz_via_fusion_returns_correct_outcome_list_length` | Off-by-one in range(C-1) |
| `test_ghz_via_fusion_4qubit_always_succeeds` | Wrong correction for any of 4 outcomes (20 runs) |
| `test_ghz_via_fusion_4qubit_all_outcomes_covered` | Correction silently broken for specific outcomes (100 runs) |
| `test_ghz_via_fusion_6qubit_k3` | Sequential decoder error for multi-boundary (20 runs, C=4) |
| `test_ghz_cluster_prep_circuits_are_independent` | Cross-cluster entanglement before fusion |

---

## Deviations from proposal

- **Test `test_ghz_cluster_prep_circuit_on_contiguous_sites_matches_ghz_circuit`**: proposal
  compared to `ghz_circuit(3)`, but `ghz_circuit` was broken by the `construct_layers` bug
  (chain CNOT reordering). Changed to verify the output state directly against |GHZ_3⟩.
  The `construct_layers` bugfix was added as a bonus — not in the proposal, but unavoidable.

- **`measure_and_collapse` order**: followed existing implementation `(Statevector, str)` as
  agreed; updated `API_VISION.md` to match.

---

## Checklist for user review

- [ ] Open `examples/devbooks/ghz_fusion.ipynb` and run all cells
- [ ] Verify cell 4 (4-qubit, 20 runs): all fidelities = 1.0
- [ ] Verify cell 5 (outcomes, 200 runs): all 4 outcomes appear, roughly uniform
- [ ] Verify cell 6 (6-qubit, 20 runs): all fidelities = 1.0
- [ ] Verify cell 3 (pre-fusion independence): Purity of each cluster rdm = 1.0
- [ ] Review `construct_layers` fix in `core/circuits.py` (lines 31–55)
