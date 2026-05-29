# Proposal: Task 17 — FH Entanglement Structure and Circuit Complexity

## Approach

Implement three pure functions in a new `applications/quantum_info.py` module, then build an exploratory notebook using `FermiHubbard1D.ground_state()` and `environment_state_prep`.

---

## Functions

### `entanglement_entropy(sv: Statevector, sites: list[int]) -> float`

1. Call `sv.rdm(sites)` → Hermitian matrix of shape `(2^|sites|, 2^|sites|)`.
2. Eigendecompose: `eigenvalues = np.linalg.eigvalsh(rdm)`.
3. Clip near-zero: `eigenvalues = np.clip(eigenvalues, 1e-15, None)`.
4. Normalise (eigenvalues may not sum to exactly 1 due to floating point): `eigenvalues /= eigenvalues.sum()`.
5. Return `float(-np.sum(eigenvalues * np.log(eigenvalues)))`.

The clip threshold of `1e-15` avoids `log(0)` while being well below any physically meaningful eigenvalue at double precision.

### `entanglement_profile(sv: Statevector) -> np.ndarray`

Loop over bipartition cuts `k = 0, …, n-2`:

```python
sites_A = list(range(k + 1))
S[k] = entanglement_entropy(sv, sites_A)
```

Returns `np.ndarray` of length `n - 1`. Delegates entirely to `entanglement_entropy`; no separate rdm logic.

### `entanglement_depth(sv: Statevector) -> int`

1. Compute `profile = entanglement_profile(sv)`.
2. Return `int(np.ceil(profile.max() / np.log(2)))`.

Rationale: a depth-`d` nearest-neighbour brickwall on a 1D chain can produce at most `d * log(2)` bits of bipartite entanglement across any cut (each layer contributes at most one ebit per cut). This is the standard linear-depth lower bound from entanglement area laws.

---

## Tests (`tests/test_applications/test_quantum_info.py`)

| Test | State | Expected |
|---|---|---|
| `test_entanglement_entropy_product_state` | `\|00…0⟩` (bitstring), cut on `[0]` | `S = 0` |
| `test_entanglement_entropy_bell_state` | Bell state via `ghz_circuit(2)` applied to `\|00⟩`, cut on `[0]` | `S = log(2)` |
| `test_entanglement_profile_length` | any state, `n` qubits | `len(profile) == n - 1` |
| `test_entanglement_depth_product` | `\|00…0⟩` | `depth == 0` |
| `test_entanglement_depth_ghz` | GHZ on 4 qubits | `depth >= 1` |

Bell state construction: `Statevector(bitstring="00").apply(ghz_circuit(2))`.

---

## Notebook structure (`examples/devbooks/fh_entanglement.ipynb`)

1. **Entanglement profile** — `S(k)` for stacked and interleaved, `num_fermions = 2, 3, 4`, at half-filling (`n_electrons = num_fermions`). Two side-by-side subplots (one per ordering).

2. **Scaling of max bipartite entanglement** — scatter/line of `max(S(k))` vs `num_fermions ∈ {2, 3, 4, 5, 6}`, both orderings on the same axes. Expect roughly linear growth (volume law would be surprising; area-law-ish is expected for gapped systems, but FH can be gapless at half-filling).

3. **Entanglement depth** — table or bar chart of `entanglement_depth(gs)` vs `num_fermions` for both orderings.

4. **Preparation depth** — for `num_fermions = 2, 3, 4`: sweep brickwall depth `d = 1, 2, …`. For each `d`, build `skeleton = brickwall_skeleton(2 * num_fermions) * d` and call `environment_state_prep(target=gs, init_state=Statevector(bitstring="0"*n), skeleton=skeleton, context=RunContext(max_iter=300))`. The function internally initialises a `two_local_circuit(skeleton)` of generic 2-qubit unitaries and optimises via SVD — no parameterised gates (RZZ etc.) are involved. Record `fidelity = 1 - cost_list[-1]` at convergence. Preparation depth is the minimum `d` at which fidelity ≥ 0.99. Stop sweeping `d` at 12 or when threshold is reached.

5. **Ordering comparison** — prose + numerical comparison: which ordering achieves lower `entanglement_depth`? Discuss JW string lengths and why interleaved sometimes localises correlations better for nearest-neighbour devices.

---

## Edge cases and uncertainties

- **`rdm` convention**: `sv.rdm(sites)` traces over the complement. I'm assuming `sites` is the subsystem we *keep*, consistent with how `partial_overlap` works in `statevector.py`. Will verify in tests.

- **Normalisation after clipping**: clipping can cause eigenvalues to not sum to exactly 1. Renormalising before taking entropy is numerically safer and changes the result by ε ≈ machine precision for physical states.

- **`entanglement_entropy` with `sites = []` or `sites = all`**: both should return 0 (pure state entropy). The rdm is a scalar 1 or 1, `log(1) = 0`. Will not add a special case; the formula handles it naturally.

- **Section 4 (preparation depth)**: `environment_state_prep` with `skeleton` calls `two_local_circuit(skeleton)` internally, which creates random Haar-distributed 4×4 unitaries — generic, not parameterised. The SVD environment update always returns a generic unitary, so hardware-native gates (RZZ, RX, etc.) are incompatible with this routine. Depth-`d` ansatz is `skeleton = brickwall_skeleton(n) * d`; no pre-built circuit is constructed. Preparation depth is the minimum `d` where fidelity ≥ 0.99 after sweep convergence.

---

## Questions for user before proceeding

1. `entanglement_entropy(sv, sites=[])` (empty subsystem, pure state, S=0) — guard explicitly or let the formula fall through naturally?
