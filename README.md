# Qaravan

[![PyPI version](https://img.shields.io/pypi/v/qaravan.svg)](https://pypi.org/project/qaravan/)
[![codecov](https://codecov.io/gh/alam-faisal/qaravan/branch/master/graph/badge.svg)](https://codecov.io/gh/alam-faisal/qaravan/branch/master)

**Qaravan** is a Python library for classical simulation of quantum circuits. It is designed for rapid prototyping of near-term quantum algorithms, error mitigation techniques, and mid-circuit measurement applications.

> **v0.2 is under active development.** Backends marked "In legacy" below still live under `qaravan.legacy` and are being ported to the new abstractions.

## Design

Qaravan is built around four core abstractions:

- **`Gate`** — an operation with a name, site indices, and a matrix representation. 
- **`Circuit`** — an ordered sequence of gates. Supports `+` (concatenation), `*` (repetition), `.dagger()`.
- **`State`** — the evolving data structure (statevector, density matrix, MPS, ...). Has `.expectation()`, `.sample()`, `.sample_and_collapse()`.
- **`Simulator`** — takes an initial `State`, a `Circuit`, and an optional `NoiseModel`. Compiles and evolves; returns the final `State`.

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
from qaravan.core import Circuit, PauliString
from qaravan.core.gates import H, CNOT
from qaravan.backends import Statevector

circ = Circuit([H(0), CNOT([0, 1])])
final_sv = Statevector(bitstring="00").apply(circ)
print("⟨ZZ⟩:", final_sv.expectation(PauliString("ZZ")))   # → 1.0
```

`state.apply(circ)` is shorthand for `StatevectorSimulator(circ, state).run()`.

### Noisy simulation

A single noisy `Statevector` run samples one error realization and returns one pure state. `Walkers` evolves an ensemble of independent trajectories, which represents the mixed state statistically (see [GOTCHAS](docs/GOTCHAS.md)).

```python
from qaravan.backends import Walkers
from qaravan.core.noise import DepolarizingNoise

noise = DepolarizingNoise(num_sites=2, error_rate=0.1)
walkers = Walkers(Statevector(bitstring="00"), num_walkers=2000)
noisy = walkers.apply(circ, noise_model=noise, noise_rng=0)
print("⟨ZZ⟩:", noisy.expectation(PauliString("ZZ")))
```

### Dynamic circuits (mid-circuit measurement + feed-forward)

```python
from qaravan.core import DynamicCircuit, DynamicRound

def decoder(outcome: str) -> Circuit:
    return Circuit([]) if outcome == "0" else Circuit([H(1)])

dynamic = DynamicCircuit([DynamicRound(meas_sites=[0], decoder=decoder)])
final_state, record = dynamic.execute(final_sv, measurement_rng=1)
```

### Trotter time evolution

```python
from qaravan.core import TFI, Magnetization
from qaravan.applications import trotter_circuit

ham = TFI(n=8, J=1.0, h=0.75)
step = trotter_circuit(ham, dt=0.1)
obs = Magnetization(n=8)

state = Statevector(bitstring="0" * 8)
mags = []
for _ in range(50):
    mags.append(state.expectation(obs))
    state = state.apply(step)
```

## Backends planned

| Backend | State type | Notes |
|---|---|---|
| `StatevectorSimulator` | `Statevector` | Complete |
| `DensityMatrixSimulator` | `DensityMatrix` | In legacy |
| `MPSSimulator` | `MPS` | In legacy |
| `MPDOSimulator` | `MPDO` | In legacy |
| `MonteCarloSimulator` | `Walkers` | In development |
| `CliffordSimulator` | `StabilizerTableau` | In development |
| `MatchgateSimulator` | `GaussianState` | In legacy |
| `PauliPropagationSimulator` | `PauliSum` | In development |
