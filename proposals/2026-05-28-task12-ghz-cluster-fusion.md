# Proposal: Task 12 — GHZ state prep via cluster fusion

## Approach

Implement the canonical measurement-based GHZ state preparation: prepare independent GHZ
clusters, fuse neighboring clusters with Bell measurements, and apply Pauli corrections based
on outcomes. This validates the DQC loop from `API_VISION.md` using existing infrastructure
(`measure_and_collapse` and `StatevectorSimulator` are already in place) and adds three
circuit-library primitives.

No new abstractions (those come in Task 13). The DQC loop runs raw in `ghz_via_fusion`.

---

## Protocol derivation

### Setup (4-qubit case, concrete acceptance test)

Six physical qubits. Cluster 1 on {0,1,2}, cluster 2 on {3,4,5}. In both clusters, qubit 0
(of the cluster) is the root. The `ghz_circuit(3)` on shifted sites gives:

```
Cluster 1: H(0), CNOT(0,1), CNOT(0,2)  →  (|000⟩₀₁₂ + |111⟩₀₁₂)/√2
Cluster 2: H(3), CNOT(3,4), CNOT(3,5)  →  (|000⟩₃₄₅ + |111⟩₃₄₅)/√2
```

Fuse qubit 2 (last leaf of cluster 1) with qubit 3 (root of cluster 2).
Bell measurement pre-rotation: CNOT(2→3), H(2). Measure sites {2, 3}. Outcome string "ab"
where a = measurement result of qubit 2, b = result of qubit 3.

Joint state before measurement (after Bell circuit):

```
|ψ⟩ = (1/2)(|000⟩₀₁₂ + |111⟩₀₁₂)(|000⟩₃₄₅ + |111⟩₃₄₅)
```

Working through CNOT(2→3), H(2) and collecting by (a, b) on qubits 2,3:

| outcome "ab" | remaining state on {0,1,4,5}       | correction               |
|:---:|:---:|:---:|
| "00"          | `(|0000⟩ + |1111⟩)/√2` = GHZ₄ ✓   | none                     |
| "01"          | `(|0011⟩ + |1100⟩)/√2`             | X on qubits 4, 5         |
| "10"          | `(|0000⟩ - |1111⟩)/√2`             | Z on qubit 0             |
| "11"          | `(|0011⟩ - |1100⟩)/√2`             | Z on qubit 0; X on 4, 5  |

The X correction targets **all kept qubits of cluster 2** ({4, 5} here), not a single qubit.
The Z correction is always single-qubit. Both are O(k-1) gates in depth-1.

### General case: C clusters of size k

```
n_final = C*(k-2) + 2  →  C = (n_final - 2) / (k - 2),  valid for k > 2
total_qubits = C * k
```

Site layout for cluster j (0-indexed):
- All sites: [j*k, ..., (j+1)*k − 1]
- Root (left fusion qubit, if j > 0): j*k
- Right fusion qubit (if j < C−1): (j+1)*k − 1
- Kept sites:
  - Cluster 0 (first): [0, ..., k−2]  (k−1 qubits)
  - Interior cluster j: [j*k+1, ..., (j+1)*k−2]  (k−2 qubits)
  - Cluster C−1 (last): [(C-1)*k+1, ..., C*k−1]  (k−1 qubits)

At each boundary j (between cluster j and j+1):
- Left fusion qubit: q_L = (j+1)*k − 1
- Right fusion qubit: q_R = (j+1)*k
- Bell circuit: CNOT(q_L→q_R), H(q_L). Measure {q_L, q_R}. Outcome "a_j b_j".
- Correction (applied immediately — sequential decoder):
  - a_j = 1 → Z on qubit 0 (any fixed anchor qubit in the chain)
  - b_j = 1 → X on every kept qubit of clusters j+1 through C−1

**Why no exponential lookup table:** each boundary's correction is computed and applied
independently, before the next fusion starts. The state is a clean GHZ entering each fusion.
Multiple Z corrections to qubit 0 accumulate as XOR (ZZ=I), which is automatically handled
by sequential application. The total correction work: O(C) decisions + O(n_final) single-qubit
X gates (depth-1) + O(C) single-qubit Z gates. All O(n_final), not O(4^C).

---

## Functions

### `applications/circuit_library.py`

```python
def bell_basis_circuit(a: int, b: int, num_sites: int) -> Circuit:
    """CNOT(a→b) then H(a); pre-rotation before Bell-basis measurement of qubits a, b."""
```

```python
def ghz_cluster_prep_circuit(cluster_sites: list[int], num_sites: int) -> Circuit:
    """GHZ state on cluster_sites in a num_sites-qubit register.

    H(cluster_sites[0]), CNOT(cluster_sites[0], s) for s in cluster_sites[1:].
    Generalises ghz_circuit to non-contiguous or offset sites.
    """
```

### `applications/compilation.py`

```python
def ghz_via_fusion(n: int, k: int = 3) -> tuple[Statevector, list[str]]:
    """Prepare n-qubit GHZ via mid-circuit-measurement fusion of k-qubit clusters.

    Returns (final_sv, outcomes) where final_sv is on C*k physical qubits
    (fusion qubits collapsed to definite states) and outcomes[j] is the 2-char
    outcome string for boundary j.
    Requires k > 2 and (n - 2) % (k - 2) == 0.
    """
```

Internal structure:
```python
C = (n - 2) // (k - 2)
total_qubits = C * k

# Step 1 — prepare all clusters
pre_circ = Circuit(
    [gate for j in range(C)
     for gate in ghz_cluster_prep_circuit(list(range(j*k, (j+1)*k)), total_qubits).gates],
    num_sites=total_qubits
)
sv = StatevectorSimulator(pre_circ, Statevector(bitstring="0" * total_qubits)).run()

# Step 2 — fuse boundaries sequentially
outcomes = []
for j in range(C - 1):
    q_L = (j + 1) * k - 1
    q_R = (j + 1) * k
    sv = StatevectorSimulator(bell_basis_circuit(q_L, q_R, total_qubits), sv).run()
    sv, outcome_str = sv.measure_and_collapse([q_L, q_R])
    outcomes.append(outcome_str)

    # Step 3 — sequential correction
    a, b = int(outcome_str[0]), int(outcome_str[1])
    right_kept = [
        site
        for jj in range(j + 1, C)
        for site in _cluster_kept_sites(jj, k, C)
    ]
    correction_gates = []
    if b == 1:
        correction_gates += [X(s) for s in right_kept]
    if a == 1:
        correction_gates += [Z(0)]
    if correction_gates:
        sv = StatevectorSimulator(Circuit(correction_gates, num_sites=total_qubits), sv).run()

return sv, outcomes
```

Private helper (module-level, no tests required):
```python
def _cluster_kept_sites(j: int, k: int, C: int) -> list[int]:
    """Return kept (non-fusion) site indices for cluster j."""
```

---

## Tests (outside-in)

### `tests/test_applications/test_circuit_library.py`

```
test_bell_basis_circuit_rotates_to_bell_basis
  Apply bell_basis_circuit(0, 1, 2) to each of the 4 computational basis states.
  Verify the resulting statevectors are proportional to the 4 Bell states.
  Catches: CNOT/H order, correct site assignment.
  Does NOT catch: wrong qubit labeling in > 2-qubit systems.

test_ghz_cluster_prep_circuit_on_contiguous_sites_matches_ghz_circuit
  cluster_sites=[0,1,2], num_sites=3 → same circuit as ghz_circuit(3).
  Catches: implementation bugs in ghz_cluster_prep_circuit.

test_ghz_cluster_prep_circuit_on_offset_sites
  cluster_sites=[3,4,5], num_sites=6 → Statevector should be
  |0⟩^⊗3 ⊗ (|000⟩+|111⟩)/√2 on qubits 3,4,5.
  Catches: site offset errors.
```

### `tests/test_applications/test_compilation.py`

```
test_ghz_via_fusion_4qubit_always_succeeds
  Run ghz_via_fusion(4, k=3) 20 times (different random seeds via np.random.default_rng).
  Each run: compute rdm([0,1,4,5]) and verify fidelity with |GHZ₄⟩⟨GHZ₄| = 1.0.
  Catches: wrong correction for any of the 4 outcomes.
  Does NOT catch: bugs that only manifest for specific random seeds.

test_ghz_via_fusion_4qubit_all_outcomes_covered
  Run 100 times; assert all 4 outcome strings {"00","01","10","11"} appear.
  Catches: correction logic that silently breaks for specific outcomes.

test_ghz_via_fusion_raises_for_bad_n_k
  ghz_via_fusion(5, k=3) raises ValueError (5-2=3 not divisible by 3-2=1... wait 3%1=0).
  Actually: ghz_via_fusion(5, k=3): C=(5-2)/1=3, OK. ghz_via_fusion(5, k=4): C=(5-2)/2=1.5, not int → raises.
  Catches: missing input validation.

test_ghz_via_fusion_6qubit_k3
  n=6, k=3 → C=4 clusters, 3 fusions. 20 runs, all fidelity 1.0 on kept sites.
  Catches: sequential decoder error for multi-boundary case.

test_ghz_via_fusion_returns_correct_outcome_list_length
  len(outcomes) == C - 1 == (n - 2) // (k - 2) - 1.
  Catches: loop bounds bug.

test_ghz_cluster_prep_circuits_are_independent
  For 2 clusters on disjoint sites, running both prep circuits and tracing out one cluster
  yields the maximally mixed state on the other — they're unentangled before fusion.
  Catches: cross-cluster entanglement before fusion (would cause wrong correction logic).
```

---

## Edge cases and complications

**`measure_and_collapse` return order**: The current implementation returns
`(new_sv, outcome_str)`, but `API_VISION.md` shows `outcome, sv = sv.measure_and_collapse(...)`.
The order is swapped. I will follow the existing implementation's return order and note
the API inconsistency without fixing it (fixing it would break Task 7's simulator code). Flag
for cleanup in a future task.

**Qubit 0 as Z-anchor**: The Z correction always targets qubit 0. For the 4-qubit case,
qubit 0 is a kept qubit of cluster 1. For the general case, qubit 0 is the root of cluster 1
(a kept qubit). This is always valid since cluster 1 always keeps its root.

**`ghz_via_fusion` returns all C*k physical qubits**, not just the n kept ones. The fusion
qubits remain in the Statevector in their collapsed states. The caller accesses kept qubits
via `sv.rdm(kept_sites)` or similar. Returning a reduced state would require knowing the
kept sites externally; keeping all sites is consistent with `measure_and_collapse` semantics.

**k=2 is excluded**: (n-2)/(k-2) is undefined. k=2 clusters (Bell pairs) fuse to give
larger GHZ states but the formula breaks — this is a degenerate case with a different protocol.
k=2 gets a clear ValueError.

---

## Things I am unsure about

**Decoder Z-correction for Z₀ vs any qubit**: The derivation says Z on "any one qubit."
Choosing qubit 0 is arbitrary. If future tasks need to compose `ghz_via_fusion` with further
operations, the anchor choice may matter. For now qubit 0 is fine — note in code comment.

**Sequential vs batched corrections**: The proposal applies corrections immediately after each
fusion. An alternative is to accumulate all corrections across all fusions and apply once at
the end. The batched approach reduces circuit depth (all Z corrections XOR to ≤ 1; all X
corrections may partially cancel too). For n_final qubits this could halve the gate count.
The sequential approach is simpler and correct. Prefer simplicity here; optimize later if needed.

**`ghz_cluster_prep_circuit` name**: possibly just `ghz_circuit_on_sites` to clarify the
relationship to the existing `ghz_circuit`. Will name it during implementation if the longer
name is clearer.
