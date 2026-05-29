"""Tests for HamiltonianTerm, Hamiltonian, TFI, Heisenberg1D."""

import numpy as np
import pytest
from scipy.linalg import expm

from qaravan.core.hamiltonians import (
    Heisenberg1D,
    TFI,
    _embed_pauli_string,
)
from qaravan.core.observables import PauliString


# ---------------------------------------------------------------------------
# _embed_pauli_string
# ---------------------------------------------------------------------------


def test_embed_pauli_string_interior_sites():
    ps = PauliString("ZZ", -1.0)
    embedded = _embed_pauli_string(ps, [1, 2], 4)
    assert embedded.string == "IZZI"
    assert embedded.coeff == -1.0


def test_embed_pauli_string_boundary_site():
    ps = PauliString("X", -0.5)
    embedded = _embed_pauli_string(ps, [0], 3)
    assert embedded.string == "XII"
    assert embedded.coeff == -0.5


def test_embed_pauli_string_last_site():
    ps = PauliString("Z", 2.0)
    embedded = _embed_pauli_string(ps, [3], 4)
    assert embedded.string == "IIIZ"
    assert embedded.coeff == 2.0


# ---------------------------------------------------------------------------
# Generic Hamiltonian — hermiticity
# ---------------------------------------------------------------------------


def test_tfi_as_matrix_is_hermitian():
    rng = np.random.default_rng(42)
    J, h = float(rng.uniform(0.5, 2.0)), float(rng.uniform(0.1, 1.0))
    tfi = TFI(4, J=J, h=h)
    mat = tfi.as_matrix()
    assert np.allclose(mat, mat.conj().T, atol=1e-12)


def test_heisenberg_as_matrix_is_hermitian():
    heis = Heisenberg1D(4, J=1.0, h=0.3)
    mat = heis.as_matrix()
    assert np.allclose(mat, mat.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# TFI ground state
# ---------------------------------------------------------------------------


def test_tfi_ground_state_energy_h0():
    # h=0, J=1, open BC: ferromagnet ground state, E = -(n-1)*J
    for n in [2, 3, 4]:
        tfi = TFI(n, J=1.0, h=0.0, periodic=False)
        sv = tfi.ground_state()
        energy = sv.expectation(tfi.as_observable()).real
        expected = -(n - 1)
        assert abs(energy - expected) < 1e-9, (
            f"n={n}: got {energy}, expected {expected}"
        )


def test_tfi_ground_state_against_exact_diag():
    # |⟨computed | exact⟩|² = 1 up to global phase
    for n in range(3, 6):
        tfi = TFI(n, J=1.0, h=0.5, periodic=False)
        sv = tfi.ground_state()
        evals, evecs = np.linalg.eigh(tfi.as_matrix())
        exact_gs = evecs[:, 0]
        fidelity = abs(np.dot(sv.to_array().conj(), exact_gs)) ** 2
        assert fidelity > 1 - 1e-10, f"n={n}: fidelity={fidelity}"


# ---------------------------------------------------------------------------
# TFI Trotter
# ---------------------------------------------------------------------------


def test_tfi_trotter_raises_for_n2():
    tfi = TFI(2, J=1.0, h=0.5)
    with pytest.raises(ValueError):
        tfi.trotter_circuit(0.1)


def test_tfi_trotter_circuit_order2_vs_propagator():
    # Order-2 error per step ~ O(dt^3); dt=0.005 gives error ~ 1e-7
    n, J, h, dt = 4, 1.0, 0.5, 0.005
    tfi = TFI(n, J=J, h=h, periodic=False)
    circ = tfi.trotter_circuit(dt, order=2)
    trotter_mat = circ.to_matrix()
    exact_mat = expm(-1j * dt * tfi.as_matrix())
    frob_err = np.linalg.norm(trotter_mat - exact_mat, ord="fro")
    assert frob_err < 1e-5, f"Frobenius error {frob_err:.2e}"


def test_tfi_trotter_error_scaling_order1():
    # log-log slope of Frobenius error vs dt should be ~2 for order-1
    n, J, h = 4, 1.0, 0.5
    tfi = TFI(n, J=J, h=h, periodic=False)
    dt_values = [0.1, 0.05, 0.02, 0.01]
    errors = []
    for dt in dt_values:
        circ = tfi.trotter_circuit(dt, order=1)
        errors.append(
            np.linalg.norm(
                circ.to_matrix() - expm(-1j * dt * tfi.as_matrix()), ord="fro"
            )
        )
    slope = np.polyfit(np.log(dt_values), np.log(errors), 1)[0]
    assert 1.8 < slope < 2.2, f"Order-1 slope={slope:.3f} (expected ~2)"


def test_tfi_trotter_error_scaling_order2():
    # log-log slope should be ~3 for order-2
    n, J, h = 4, 1.0, 0.5
    tfi = TFI(n, J=J, h=h, periodic=False)
    dt_values = [0.1, 0.05, 0.02, 0.01]
    errors = []
    for dt in dt_values:
        circ = tfi.trotter_circuit(dt, order=2)
        errors.append(
            np.linalg.norm(
                circ.to_matrix() - expm(-1j * dt * tfi.as_matrix()), ord="fro"
            )
        )
    slope = np.polyfit(np.log(dt_values), np.log(errors), 1)[0]
    assert 2.7 < slope < 3.3, f"Order-2 slope={slope:.3f} (expected ~3)"


# ---------------------------------------------------------------------------
# Heisenberg1D ground state
# ---------------------------------------------------------------------------


def test_heisenberg_ground_state_energy_j0():
    # J=0, h=1: H = -h Σ Z_i; |0...0⟩ is ground state with E = -h*n
    for n in [2, 3, 4]:
        heis = Heisenberg1D(n, J=0.0, h=1.0)
        sv = heis.ground_state()
        energy = sv.expectation(heis.as_observable()).real
        expected = -float(n)
        assert abs(energy - expected) < 1e-9, (
            f"n={n}: got {energy}, expected {expected}"
        )


def test_heisenberg_ground_state_against_exact_diag():
    for n in range(3, 6):
        heis = Heisenberg1D(n, J=1.0, h=0.5)
        sv = heis.ground_state()
        evals, evecs = np.linalg.eigh(heis.as_matrix())
        exact_gs = evecs[:, 0]
        fidelity = abs(np.dot(sv.to_array().conj(), exact_gs)) ** 2
        assert fidelity > 1 - 1e-10, f"n={n}: fidelity={fidelity}"


# ---------------------------------------------------------------------------
# Heisenberg1D Trotter
# ---------------------------------------------------------------------------


def test_heisenberg_trotter_raises_for_n2():
    heis = Heisenberg1D(2, J=1.0)
    with pytest.raises(ValueError):
        heis.trotter_circuit(0.1)


def test_heisenberg_trotter_vs_propagator():
    n, J, h, dt = 4, 1.0, 0.3, 0.005
    heis = Heisenberg1D(n, J=J, h=h, periodic=False)
    circ = heis.trotter_circuit(dt, order=2)
    trotter_mat = circ.to_matrix()
    exact_mat = expm(-1j * dt * heis.as_matrix())
    frob_err = np.linalg.norm(trotter_mat - exact_mat, ord="fro")
    assert frob_err < 1e-5, f"Frobenius error {frob_err:.2e}"


def test_heisenberg_xx_model_ground_energy():
    # J=1, h=0, PBC: ground energy from ground_state() matches eigh minimum
    for n in [4, 6]:
        heis = Heisenberg1D(n, J=1.0, h=0.0, periodic=True)
        evals = np.linalg.eigvalsh(heis.as_matrix())
        sv = heis.ground_state()
        energy = sv.expectation(heis.as_observable()).real
        assert abs(energy - evals[0]) < 1e-9, (
            f"n={n}: got {energy}, expected {evals[0]}"
        )
