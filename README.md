# Qaravan

![PyPI version](https://img.shields.io/pypi/v/qaravan)
[![Tests](https://github.com/alam-faisal/qaravan/actions/workflows/tests.yml/badge.svg)](https://github.com/alam-faisal/qaravan/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/alam-faisal/qaravan/branch/master/graph/badge.svg)](https://codecov.io/gh/alam-faisal/qaravan)

**Qaravan** is a Python library for simulating quantum circuits with and without noise using a variety of classical simulation techniques. It has built in GPU acceleration and autodiff support, and designed with usability and extensibility in mind.  

## Features

- **tensorQ**: statevector, density matrix, MPS, MPDO methods
- **algebraQ**: Lie algebraic, doped matchgate, doped Clifford (in development) simulators
- **Noise models**: most simulators support noise models natively
- **Arbitrary qudits**: tensorQ supports quantum systems beyong qubits
- **2D lattices**: built-in support for several commonly used 2D lattices
- **GPU acceleration + autodiff**: PyTorch backend for select simulators
- **Algorithmic tools**: algorithms for state preparation, circuit synthesis, ground state search
- **Tensor network tools**: environment-based methods

## Installation

You can install Qaravan from PyPI:

```bash
pip install qaravan
```

## Quick Start

``` 
from qaravan.tensorQ import StatevectorSim
import torch

# Create a simple 2-qubit Hadamard-CNOT circuit
circ = Circuit([H(0), CNOT([0,1])])

# Simulate it
sim = StatevectorSim(circ, backend="torch")
sim.run()
print("Statevector:", sim.get_statevector())
```

## Examples

Check out the `examples/` directory for notebooks on:
- autodiff-based state preparation 
- environment-based state preparation
- ground state search on 1D lattice
- noisy simulation of a quantum algorithm

## Roadmap 

- ✅ **tensorQ**: Full tensor network-based simulators with noise modeling
- ✅ **PyTorch integration**: GPU acceleration + autodiff
- ✅ **algebraQ**: Doped matchgate simulator
- 🔧 **algebraQ**: Lie algebraic simulator 
- ⬜ **algebraQ**: Doped Clifford simulator 
- ⬜ **pathQ**: Pauli path propagation 
- 🔧 **pathQ**: Noise sampling-based simulator