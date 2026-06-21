"""Tests for FermiHubbard1D with Jordan-Wigner encoding."""

import numpy as np
from scipy.linalg import expm

from qaravan.core.hamiltonians import FermiHubbard1D


# ---------------------------------------------------------------------------
# Hermiticity
# ---------------------------------------------------------------------------


def test_fh_as_matrix_is_hermitian_stacked():
    rng = np.random.default_rng(0)
    t, U, mu = (
        float(rng.uniform(0.5, 2.0)),
        float(rng.uniform(0, 3.0)),
        float(rng.uniform(-0.5, 0.5)),
    )
    fh = FermiHubbard1D(3, t=t, U=U, mu=mu, ordering="stacked")
    H = fh.as_matrix()
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_fh_as_matrix_is_hermitian_interleaved():
    rng = np.random.default_rng(1)
    t, U, mu = (
        float(rng.uniform(0.5, 2.0)),
        float(rng.uniform(0, 3.0)),
        float(rng.uniform(-0.5, 0.5)),
    )
    fh = FermiHubbard1D(3, t=t, U=U, mu=mu, ordering="interleaved")
    H = fh.as_matrix()
    assert np.allclose(H, H.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# Encoding agreement (sharpest correctness test — catches string bugs)
# ---------------------------------------------------------------------------


def test_fh_encoding_agreement():
    """Both orderings must give the same full spectrum for the same physical params."""
    for num_fermions in [3, 4]:
        fh_s = FermiHubbard1D(num_fermions, t=1.0, U=2.0, mu=0.5, ordering="stacked")
        fh_i = FermiHubbard1D(
            num_fermions, t=1.0, U=2.0, mu=0.5, ordering="interleaved"
        )
        E_s = np.linalg.eigvalsh(fh_s.as_matrix())[0]
        E_i = np.linalg.eigvalsh(fh_i.as_matrix())[0]
        assert abs(E_s - E_i) < 1e-10, (
            f"num_fermions={num_fermions}: stacked E0={E_s}, interleaved E0={E_i}"
        )


# ---------------------------------------------------------------------------
# Hubbard dimer analytical check
# ---------------------------------------------------------------------------


def test_fh_ground_state_dimer_exact():
    """At half-filling (n_electrons=2), E_gs = (U - sqrt(U^2 + 16t^2)) / 2 for the dimer."""
    for ordering in ["stacked", "interleaved"]:
        for t, U in [(1.0, 0.0), (1.0, 2.0), (0.5, 4.0)]:
            fh = FermiHubbard1D(num_fermions=2, t=t, U=U, mu=0.0, ordering=ordering)
            sv = fh.ground_state(n_electrons=2)
            energy = sv.expectation(fh.as_observable()).real
            expected = (U - np.sqrt(U**2 + 16 * t**2)) / 2
            assert abs(energy - expected) < 1e-10, (
                f"ordering={ordering}, t={t}, U={U}: got {energy:.8f}, expected {expected:.8f}"
            )


# ---------------------------------------------------------------------------
# Free-fermion limit
# ---------------------------------------------------------------------------


def test_fh_free_fermion_ground_energy():
    """At U=0 the FH reduces to two decoupled tight-binding chains.
    Ground energy = 2 * sum of occupied tight-binding levels.
    """
    num_fermions = 4
    t = 1.0
    # Open-chain tight-binding: ε_k = -2t cos(πk/(L+1)), k=1,...,L
    tb_evals = sorted(
        -2 * t * np.cos(np.pi * k / (num_fermions + 1))
        for k in range(1, num_fermions + 1)
    )
    n_fill = num_fermions // 2  # electrons per spin at half-filling
    expected = 2 * sum(tb_evals[:n_fill])

    for ordering in ["stacked", "interleaved"]:
        fh = FermiHubbard1D(
            num_fermions=num_fermions, t=t, U=0.0, mu=0.0, ordering=ordering
        )
        E0 = np.linalg.eigvalsh(fh.as_matrix())[0]
        assert abs(E0 - expected) < 1e-8, (
            f"ordering={ordering}: got {E0:.8f}, expected {expected:.8f}"
        )


def test_fh_free_fermion_stacked_spin_decoupling():
    """At U=0 stacked, ground state is |GS_↑⟩ ⊗ |GS_↓⟩.
    The reduced density matrix of qubits L..2L-1 (down sector) must be pure: Tr[ρ²] = 1.
    """
    num_fermions = 4
    fh = FermiHubbard1D(
        num_fermions=num_fermions, t=1.0, U=0.0, mu=0.0, ordering="stacked"
    )
    sv = fh.ground_state()
    down_sites = list(range(num_fermions, 2 * num_fermions))
    rho = sv.rdm(down_sites)
    purity = np.trace(rho @ rho).real
    assert abs(purity - 1.0) < 1e-8, f"purity={purity:.8f}; expected 1.0 (pure state)"


# ---------------------------------------------------------------------------
# Symmetry sectors: ⟨N⟩ = k for fixed-N ground state
# ---------------------------------------------------------------------------


def test_fh_electron_number_conservation():
    """ground_state(n_electrons=k) must give ⟨N⟩ = k."""
    num_fermions = 3
    num_qubits = 2 * num_fermions

    # number operator matrix: diagonal entry i = popcount(i)
    n_op = np.diag([float(bin(i).count("1")) for i in range(2**num_qubits)])

    for ordering in ["stacked", "interleaved"]:
        fh = FermiHubbard1D(
            num_fermions=num_fermions, t=1.0, U=2.0, mu=0.0, ordering=ordering
        )
        for n_elec in [2, 3, 4]:
            sv = fh.ground_state(n_electrons=n_elec)
            arr = sv.to_array()
            expectation = (arr.conj() @ n_op @ arr).real
            assert abs(expectation - n_elec) < 1e-10, (
                f"ordering={ordering}, n_elec={n_elec}: ⟨N⟩={expectation:.6f}"
            )


# ---------------------------------------------------------------------------
# Trotter: circuit vs exact propagator
# ---------------------------------------------------------------------------


def test_fh_trotter_vs_propagator_stacked():
    num_fermions, dt = 3, 0.01
    fh = FermiHubbard1D(
        num_fermions=num_fermions, t=1.0, U=2.0, mu=0.0, ordering="stacked"
    )
    circ = fh.trotter_circuit(dt, order=2)
    trotter_mat = circ.to_matrix()
    exact_mat = expm(-1j * dt * fh.as_matrix())
    frob_err = np.linalg.norm(trotter_mat - exact_mat, ord="fro")
    assert frob_err < 1e-5, f"Frobenius error {frob_err:.2e}"


def test_fh_trotter_vs_propagator_interleaved():
    num_fermions, dt = 3, 0.01
    fh = FermiHubbard1D(
        num_fermions=num_fermions, t=1.0, U=2.0, mu=0.0, ordering="interleaved"
    )
    circ = fh.trotter_circuit(dt, order=2)
    trotter_mat = circ.to_matrix()
    exact_mat = expm(-1j * dt * fh.as_matrix())
    frob_err = np.linalg.norm(trotter_mat - exact_mat, ord="fro")
    assert frob_err < 1e-5, f"Frobenius error {frob_err:.2e}"


# ---------------------------------------------------------------------------
# Periodic boundary conditions
# ---------------------------------------------------------------------------


def test_fh_pbc_l1_raises():
    """periodic=True with num_fermions=1 must raise ValueError."""
    import pytest

    with pytest.raises(ValueError, match="periodic"):
        FermiHubbard1D(1, periodic=True)


def test_fh_pbc_matrix_hermitian():
    """PBC Hamiltonian is Hermitian for both orderings at random parameters."""
    rng = np.random.default_rng(42)
    t, U, mu = (
        float(rng.uniform(0.5, 2.0)),
        float(rng.uniform(0, 3.0)),
        float(rng.uniform(-0.5, 0.5)),
    )
    for ordering in ["stacked", "interleaved"]:
        H = FermiHubbard1D(
            3, t=t, U=U, mu=mu, ordering=ordering, periodic=True
        ).as_matrix()
        assert np.allclose(H, H.conj().T, atol=1e-12), (
            f"{ordering} PBC Hamiltonian not Hermitian"
        )


def test_fh_pbc_spectrum_ordering_agreement():
    """PBC stacked and interleaved must give the same spectrum."""
    for num_fermions in [3, 4]:
        fh_s = FermiHubbard1D(
            num_fermions, t=1.0, U=2.0, mu=0.5, ordering="stacked", periodic=True
        )
        fh_i = FermiHubbard1D(
            num_fermions, t=1.0, U=2.0, mu=0.5, ordering="interleaved", periodic=True
        )
        E_s = np.linalg.eigvalsh(fh_s.as_matrix())[0]
        E_i = np.linalg.eigvalsh(fh_i.as_matrix())[0]
        assert abs(E_s - E_i) < 1e-10, (
            f"num_fermions={num_fermions}: stacked E0={E_s}, interleaved E0={E_i}"
        )


def test_fh_pbc_vs_obc_spectrum_differs():
    """PBC and OBC give different ground-state energies for L=3."""
    for ordering in ["stacked", "interleaved"]:
        fh_obc = FermiHubbard1D(3, t=1.0, U=1.0, ordering=ordering, periodic=False)
        fh_pbc = FermiHubbard1D(3, t=1.0, U=1.0, ordering=ordering, periodic=True)
        E_obc = np.linalg.eigvalsh(fh_obc.as_matrix())[0]
        E_pbc = np.linalg.eigvalsh(fh_pbc.as_matrix())[0]
        assert abs(E_obc - E_pbc) > 1e-6, (
            f"{ordering}: OBC and PBC spectra are identical (boundary term has no effect)"
        )


def test_fh_pbc_free_fermion_ground_energy():
    """At U=0 with PBC and L=4, E0 = -4t (two spins, each filling ε_0=-2t and ε_1=0).

    PBC tight-binding dispersion: ε_k = -2t cos(2πk/L), k=0..L-1.
    L=4: {-2t, 0, 2t, 0}. Half-filling (2 electrons per spin): fill -2t and 0.
    """
    L, t = 4, 1.0
    expected = -4 * t  # 2 spins × (-2t + 0)
    for ordering in ["stacked", "interleaved"]:
        fh = FermiHubbard1D(L, t=t, U=0.0, mu=0.0, ordering=ordering, periodic=True)
        E0 = np.linalg.eigvalsh(fh.as_matrix())[0]
        assert abs(E0 - expected) < 1e-8, (
            f"{ordering}: got E0={E0:.8f}, expected {expected:.8f}"
        )


def test_fh_trotter_error_scaling_order1():
    """Log-log slope of Frobenius error vs dt should be ~2 for order-1 Trotter."""
    num_fermions = 3
    dt_values = [0.1, 0.05, 0.02, 0.01]
    for ordering in ["stacked", "interleaved"]:
        fh = FermiHubbard1D(
            num_fermions=num_fermions, t=1.0, U=2.0, mu=0.0, ordering=ordering
        )
        errors = []
        for dt in dt_values:
            circ = fh.trotter_circuit(dt, order=1)
            errors.append(
                np.linalg.norm(
                    circ.to_matrix() - expm(-1j * dt * fh.as_matrix()), ord="fro"
                )
            )
        slope = np.polyfit(np.log(dt_values), np.log(errors), 1)[0]
        assert 1.8 < slope < 2.2, (
            f"ordering={ordering}: order-1 slope={slope:.3f} (expected ~2)"
        )
