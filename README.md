# Qaravan

![PyPI version](https://img.shields.io/pypi/v/qaravan)
[![Tests](https://github.com/alam-faisal/qaravan/actions/workflows/tests.yml/badge.svg)](https://github.com/alam-faisal/qaravan/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/alam-faisal/qaravan/branch/main/graph/badge.svg)](https://codecov.io/gh/alam-faisal/qaravan)



**Qaravan** is a Python library for simulating quantum circuits with and without noise using a variety of classical simulation techniques. 

## Overview

Qaravan is organized into two main submodules

**tensorQ**: contains simulators based on tensor contraction: statevector, density matrix, matrix product states and matrix product density operators

**algebraQ**: contains simulators based on algebraic methods: Clifford, matchgate, Pauli path propagation and Lie algebraic simulation (in development)

## Installation

You can install Qaravan from PyPI:

```bash
pip install qaravan