# Qaravan

[![codecov](https://codecov.io/gh/alam-faisal/qaravan/branch/rewrite%2Fv0.2/graph/badge.svg)](https://codecov.io/gh/alam-faisal/qaravan/branch/rewrite%2Fv0.2)

**Qaravan** is a Python library for classical simulation of quantum circuits. It is designed for rapid prototyping of near-term quantum algorithms, error mitigation techniques, and mid-circuit measurement applications.

> **v0.2 is under active development.** The API described here reflects the target design. Legacy v0.1 code is preserved under `qaravan.legacy` for reference.

## Design

Qaravan is built around four core abstractions:

- **`Gate`** — an operation with a name, site indices, and a matrix representation. Backend-agnostic.
- **`Circuit`** — an ordered sequence of gates. Supports `+` (concatenation), `*` (repetition), `.dagger()`.
- **`State`** — the evolving data structure (statevector, density matrix, MPS, ...). Owns `.expectation()`, `.sample()`, `.sample_and_collapse()`.
- **`Simulator`** — takes an initial `State`, a `Circuit`, and an optional `NoiseModel`. Compiles and evolves; returns the final `State`.

The `State`/`Simulator` split is the key design choice: the returned `State` is independent of the simulator, which makes dynamic-circuit and feed-forward workflows natural.

## Installation

Install from PyPI:

```bash
uv add qaravan
```

For development (clone the repo first):

```bash
uv sync --group dev
```

## Usage

### Noiseless simulation

```python
from qaravan.core.gates import H, CNOT
from qaravan.core.circuits import Circuit
from qaravan.core.observables import PauliString
from qaravan.backends.statevector import Statevector, StatevectorSimulator

circ = Circuit([H(0), CNOT([0, 1])])
obs = PauliString("ZZ")

sv = Statevector(bitstring="00")
final_sv = StatevectorSimulator(circ, sv).run()
print("⟨ZZ⟩:", final_sv.expectation(obs))   # → 1.0
```

### Noisy simulation

```python
from qaravan.core.noise import ThermalNoise
from qaravan.backends.density_matrix import DensityMatrix, DensityMatrixSimulator

nm = ThermalNoise(t1=100, t2=75, t_q1=0.04, t_q2=0.5)
dm = DensityMatrix(bitstring="00")
final_dm = DensityMatrixSimulator(circ, dm, nm).run()
print("⟨ZZ⟩:", final_dm.expectation(obs))
```

### Dynamic circuits (mid-circuit measurement + feed-forward)

```python
from qaravan.backends.statevector import Statevector, StatevectorSimulator

sv = Statevector(bitstring="0" * n)
for round in range(n_rounds):
    outcome, sv = sv.sample_and_collapse(meas_sites)
    correction = decoder(outcome, round)
    sv = StatevectorSimulator(correction, sv).run()
```

### Trotter time evolution

```python
from qaravan.core.hamiltonians import TFI
from qaravan.core.observables import Magnetization
from qaravan.backends.density_matrix import DensityMatrix, DensityMatrixSimulator

ham = TFI(n=8, j=1.0, h=0.75)
nm = ThermalNoise(t1=100, t2=75, t_q1=0.04, t_q2=0.5)
obs = Magnetization(n=8)

circ = ham.trotter_circuit(step_size=0.1)
dm = DensityMatrix(bitstring="0" * 8)
exp_list = []
for _ in range(500):
    exp_list.append(dm.expectation(obs))
    dm = DensityMatrixSimulator(circ, dm, nm).run()
```

## Backends

| Backend | State type | Noise | Notes |
|---|---|---|---|
| `StatevectorSimulator` | `Statevector` | ✗ | Reference backend |
| `DensityMatrixSimulator` | `DensityMatrix` | ✓ | Full $d^{2n}$ density matrix |
| `MPSSimulator` | `MPS` | ✗ | Bond-dimension truncation |
| `MPDOSimulator` | `MPDO` | ✓ | Mixed-state MPS |
| `MonteCarloSimulator` | `Walkers` | ✓ | Trajectory sampling |
| `CliffordSimulator` | `StabilizerTableau` | ✓ | Wraps Stim |
| `MatchgateSimulator` | `GaussianState` | — | Fermionic Gaussian circuits |
| `PauliPropagationSimulator` | `PauliSum` | — | Heisenberg-picture propagation |

## Gotchas

**`Circuit.copy()` resets layers but preserves decomposition.**
`copy()` returns a new circuit with the same gate list but `layers = None`.
`construct_layers()` must be called again on the copy before iterating over layers.
Decomposition is *not* undone: `decompose()` mutates `self.gates` in place, so a copy of a decomposed circuit still has the decomposed gates.

**Qubit ordering is big-endian: qubit 0 is the most significant bit.**
Applying `H(0)` to a 2-qubit system embeds as `H ⊗ I`, not `I ⊗ H`.
In general, gate `G` on qubit `k` of an `n`-qubit system embeds as:

```
I^{⊗k} ⊗ G ⊗ I^{⊗(n−k−1)}
```

Basis state index `i` corresponds to bitstring `q0 q1 … q_{n−1}` where `i = q0·2^{n−1} + q1·2^{n−2} + … + q_{n−1}·2^0`. Qubit 0 is the leftmost qubit in ket notation. This is the opposite of Qiskit's little-endian convention, where qubit 0 is the least significant bit and `H(0)` embeds as `I ⊗ H`.

## Roadmap

- ✅ **v0.2 scaffold** — directory structure, legacy preservation, `uv` packaging
- ✅ **Core ABCs** — `Gate`, `Circuit`, `State`, `Simulator`, `Observable`, `NoiseModel`
- 🔧 **Concrete gates** — `H`, `X`, `Y`, `Z`, `CNOT`, `RX/Y/Z`, `RXX/YY/ZZ`, `Reset`, `Measure`
- ⬜ **Statevector backend** — reference implementation
- ⬜ **Density matrix backend** — first noisy backend
- ⬜ **TFI Hamiltonian** — first end-to-end Trotter workflow
- ⬜ **MPS / MPDO backends**
- ⬜ **Mid-circuit measurements** — `Measure` gate, classical registers, feed-forward
- ⬜ **Monte Carlo backend**
- ⬜ **Clifford backend** (wraps Stim)
- ⬜ **Matchgate backend**
- ⬜ **Pauli propagation backend**
