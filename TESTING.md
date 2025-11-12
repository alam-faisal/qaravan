# Testing Backlog for Qaravan v0.2.0

**Current Coverage:** 31% (20 tests)  
**Target Coverage:** 55% for v0.2.0 release  
**Last Updated:** 2025-11-12

---

## High Priority Test Files to Create

### 1. `tests/test_circuits.py` - Circuit Operations (HIGH IMPACT)
**Current Coverage:** 37% → **Target:** 60%  

**Critical tests to add:**
- [ ] Test `Circuit.__add__()` - circuit concatenation
- [ ] Test `Circuit.__mul__()` and `__rmul__()` - circuit repetition (already designed!)
- [ ] Test `Circuit.__getitem__()` - circuit slicing and indexing
- [ ] Test `Circuit.to_matrix()` - convert small circuits to unitary matrices
- [ ] Test `Circuit.dag()` - Hermitian conjugate of circuit
- [ ] Test `Circuit.copy()` - deep copy with parametric gates
- [ ] Test `Circuit.decompose()` - basis gate decomposition
- [ ] Test `Circuit.add_noise()` - noise model integration
- [ ] Test `compose_circuits()` - compose multiple circuits
- [ ] Test `circ_to_mat()` - standalone circuit to matrix conversion

---

### 2. `tests/test_mps_sim.py` - MPS Simulator Operations
**Current Coverage:** 30% (tn.py: 52%) → **Target:** 65%  

**Critical tests to add:**
- [ ] Test `MPS.apply_one_gate()` - single-qubit gate application
- [ ] Test `MPS.apply_two_gate()` - two-qubit gate application with truncation
- [ ] Test `MPS.apply_mpo()` - MPO application (if implemented)
- [ ] Test `MPSSim.run()` - full circuit simulation with bond dimension limits
- [ ] Test `MPSSim.local_expectation()` - multi-site expectation values
- [ ] Test truncation behavior - check that bond dimensions are properly managed
- [ ] Test `MPS.normalize()` - normalization after operations
- [ ] Test `MPS.apply_reset()` - mid-circuit reset on MPS (you just implemented this!)

---

### 3. `tests/test_gates.py` - Gate Classes and Operations
**Current Coverage:** 48% → **Target:** 65%  

**Critical tests to add:**
- [ ] Test `Reset` class initialization and matrix building (you just added this!)
- [ ] Test `Reset.shallow_copy()` - ensure type preservation
- [ ] Test parametric gates (`RX`, `RY`, `RZ`, `RZZ`, etc.) from `param_gates.py`
- [ ] Test custom gate creation with `Gate(name, indices, matrix)`
- [ ] Test gate composition and tensor products
- [ ] Test gate dagger/adjoint operations
- [ ] Test multi-qubit gates (Toffoli, Fredkin, etc.) if implemented
- [ ] Test gate equality and comparison

---

### 4. `tests/test_utils.py` - Utility Functions
**Current Coverage:** 26% → **Target:** 50%  

**Critical tests to add:**
- [ ] Test `measure_sv()` - statevector measurement on various states
- [ ] Test `measure_and_collapse_sv()` - measurement with collapse
- [ ] Test `shots_to_density()` - convert measurement shots to density dict
- [ ] Test `shots_to_density_vec()` - convert shots to probability vector
- [ ] Test `l2_threshold()` - statistical threshold computation
- [ ] Test `rdm_from_sv()` - reduced density matrix from statevector
- [ ] Test `string_to_sv()` - computational basis state creation
- [ ] Test `random_sv()`, `random_unitary()`, `random_hermitian_op()` - randomization
- [ ] Test `construct_unitary()` - unitary construction from input/output pairs (already tested in test_core)

---

### 5. `tests/test_density_matrix_sim.py` - Density Matrix Simulator
**Current Coverage:** 21% → **Target:** 50%  

**Critical tests to add:**
- [ ] Test `DensityMatrixSim.run()` - full circuit simulation
- [ ] Test `DensityMatrixSim.apply_gate()` - unitary gate application
- [ ] Test `DensityMatrixSim.apply_noise()` - Kraus operator noise
- [ ] Test `DensityMatrixSim.apply_reset()` - mid-circuit reset (you just implemented this!)
- [ ] Test `DensityMatrixSim.one_local_expectation()` - efficient single-site expectations (you just added this!)
- [ ] Test `DensityMatrixSim.local_expectation()` - multi-site expectations
- [ ] Test purity tracking and mixed state handling
- [ ] Test thermal noise and decoherence

---

## Medium Priority (After reaching 55%)

### 6. More StatevectorSim edge cases
- [ ] Test normalization edge cases
- [ ] Test large circuit simulation (memory/performance)
- [ ] Test measurement on entangled states
- [ ] Test error handling for invalid operations

### 7. MPDO Simulator Tests (`test_mpdo_sim.py`)
**Current Coverage:** 16%
- [ ] Basic MPDO operations
- [ ] Noise channel application
- [ ] MPDO to density matrix conversion

### 8. Initialization Functions (`test_initializations.py`)
**Current Coverage:** 30%
- [ ] Test `random_mps()`, `ti_rmps()` (already have one test for ti_rmps)
- [ ] Test `random_mpdo()`
- [ ] Test various initial state preparations

---

## Lower Priority (Post v0.2.0)

### 9. Hamiltonians and Lattices
- Hamiltonian construction and embedding
- Lattice geometries and neighbor finding
- Trotter decomposition

### 10. Noise Models
- Thermal noise Kraus operators
- Pauli noise sampling
- Custom noise channels

### 11. Applications
- Compilation utilities
- Trotter evolution accuracy

---

## Quick Commands for Testing

```bash
# Run all tests with coverage
pytest --cov=src/qaravan --cov-report=term-missing tests/

# Run specific test file
pytest tests/test_circuits.py -v

# Generate HTML coverage report
pytest --cov=src/qaravan --cov-report=html tests/
open htmlcov/index.html

# Run tests and show only missing coverage
pytest --cov=src/qaravan --cov-report=term-missing:skip-covered tests/
```

---

## Notes

- Focus on **tests 1-5** to reach 55% coverage target for v0.2.0
- Each test file should follow the style in `test_core.py` and `test_tn.py`
- Include descriptive assertions and success print statements
- Test both happy paths and edge cases
- Prioritize tests for features shown in examples/ and documentation