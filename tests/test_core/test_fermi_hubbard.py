"""Tests for FermiHubbard1D with Jordan-Wigner encoding."""

import numpy as np
import pytest
from scipy.linalg import expm
from qaravan.applications.algorithms import ground_state, trotter_circuit
from qaravan.core.fermions import FermionicEncoding
from qaravan.core.hamiltonians import FermiHubbard1D, interleaved_mode_order


def _fh_encoding(num_fermionic_sites: int, ordering: str) -> FermionicEncoding:
    """Test-only helper: the two named layouts, as FermionicEncodings."""
    mode_order = (
        None if ordering == "stacked" else interleaved_mode_order(num_fermionic_sites)
    )
    return FermionicEncoding(
        "jordan_wigner", num_modes=2 * num_fermionic_sites, mode_order=mode_order
    )


# ---------------------------------------------------------------------------
# Hermiticity
# ---------------------------------------------------------------------------


def test_fh_to_matrix_is_hermitian_stacked():
    rng = np.random.default_rng(0)
    t, U, mu = (
        float(rng.uniform(0.5, 2.0)),
        float(rng.uniform(0, 3.0)),
        float(rng.uniform(-0.5, 0.5)),
    )
    fh = FermiHubbard1D(3, t=t, U=U, mu=mu)
    H = fh.encode(_fh_encoding(3, "stacked")).to_matrix()
    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_fh_to_matrix_is_hermitian_interleaved():
    rng = np.random.default_rng(1)
    t, U, mu = (
        float(rng.uniform(0.5, 2.0)),
        float(rng.uniform(0, 3.0)),
        float(rng.uniform(-0.5, 0.5)),
    )
    fh = FermiHubbard1D(3, t=t, U=U, mu=mu)
    H = fh.encode(_fh_encoding(3, "interleaved")).to_matrix()
    assert np.allclose(H, H.conj().T, atol=1e-12)


# ---------------------------------------------------------------------------
# Encoding agreement (sharpest correctness test — catches string bugs)
# ---------------------------------------------------------------------------


def test_fh_encoding_agreement():
    """Both orderings must give the same full spectrum for the same physical params."""
    for num_fermionic_sites in [3, 4]:
        fh = FermiHubbard1D(num_fermionic_sites, t=1.0, U=2.0, mu=0.5)
        E_s = np.linalg.eigvalsh(
            fh.encode(_fh_encoding(num_fermionic_sites, "stacked")).to_matrix()
        )[0]
        E_i = np.linalg.eigvalsh(
            fh.encode(_fh_encoding(num_fermionic_sites, "interleaved")).to_matrix()
        )[0]
        assert abs(E_s - E_i) < 1e-10, (
            f"num_fermionic_sites={num_fermionic_sites}: stacked E0={E_s}, interleaved E0={E_i}"
        )


# ---------------------------------------------------------------------------
# Hubbard dimer analytical check
# ---------------------------------------------------------------------------


def test_fh_ground_state_dimer_exact():
    """At half-filling (fixed_particle_number=2), E_gs = (U - sqrt(U^2 + 16t^2)) / 2 for the dimer."""
    for ordering in ["stacked", "interleaved"]:
        for t, U in [(1.0, 0.0), (1.0, 2.0), (0.5, 4.0)]:
            ham = FermiHubbard1D(num_fermionic_sites=2, t=t, U=U, mu=0.0).encode(
                _fh_encoding(2, ordering)
            )
            sv = ground_state(ham, fixed_particle_number=2)
            energy = sv.expectation(ham.as_observable()).real
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
    num_fermionic_sites = 4
    t = 1.0
    # Open-chain tight-binding: ε_k = -2t cos(πk/(L+1)), k=1,...,L
    tb_evals = sorted(
        -2 * t * np.cos(np.pi * k / (num_fermionic_sites + 1))
        for k in range(1, num_fermionic_sites + 1)
    )
    n_fill = num_fermionic_sites // 2  # electrons per spin at half-filling
    expected = 2 * sum(tb_evals[:n_fill])

    for ordering in ["stacked", "interleaved"]:
        fh = FermiHubbard1D(num_fermionic_sites=num_fermionic_sites, t=t, U=0.0, mu=0.0)
        H = fh.encode(_fh_encoding(num_fermionic_sites, ordering)).to_matrix()
        E0 = np.linalg.eigvalsh(H)[0]
        assert abs(E0 - expected) < 1e-8, (
            f"ordering={ordering}: got {E0:.8f}, expected {expected:.8f}"
        )


def test_fh_free_fermion_stacked_spin_decoupling():
    """At U=0 stacked, ground state is |GS_↑⟩ ⊗ |GS_↓⟩.
    The reduced density matrix of qubits L..2L-1 (down sector) must be pure: Tr[ρ²] = 1.
    """
    num_fermionic_sites = 4
    fh = FermiHubbard1D(num_fermionic_sites=num_fermionic_sites, t=1.0, U=0.0, mu=0.0)
    ham = fh.encode(_fh_encoding(num_fermionic_sites, "stacked"))
    sv = ground_state(ham)
    down_sites = list(range(num_fermionic_sites, 2 * num_fermionic_sites))
    rho = sv.rdm(down_sites)
    purity = np.trace(rho @ rho).real
    assert abs(purity - 1.0) < 1e-8, f"purity={purity:.8f}; expected 1.0 (pure state)"


# ---------------------------------------------------------------------------
# Symmetry sectors: ⟨N⟩ = k for fixed-N ground state
# ---------------------------------------------------------------------------


def test_fh_electron_number_conservation():
    """ground_state(fixed_particle_number=k) must give ⟨N⟩ = k."""
    num_fermionic_sites = 3
    num_qubits = 2 * num_fermionic_sites

    # number operator matrix: diagonal entry i = popcount(i)
    n_op = np.diag([float(bin(i).count("1")) for i in range(2**num_qubits)])

    for ordering in ["stacked", "interleaved"]:
        fh = FermiHubbard1D(
            num_fermionic_sites=num_fermionic_sites, t=1.0, U=2.0, mu=0.0
        )
        ham = fh.encode(_fh_encoding(num_fermionic_sites, ordering))
        for n_elec in [2, 3, 4]:
            sv = ground_state(ham, fixed_particle_number=n_elec)
            arr = sv.to_array()
            expectation = (arr.conj() @ n_op @ arr).real
            assert abs(expectation - n_elec) < 1e-10, (
                f"ordering={ordering}, n_elec={n_elec}: ⟨N⟩={expectation:.6f}"
            )


# ---------------------------------------------------------------------------
# Trotter: circuit vs exact propagator
# ---------------------------------------------------------------------------


def test_fh_trotter_vs_propagator_stacked():
    num_fermionic_sites, dt = 3, 0.01
    fh = FermiHubbard1D(num_fermionic_sites=num_fermionic_sites, t=1.0, U=2.0, mu=0.0)
    ham = fh.encode(_fh_encoding(num_fermionic_sites, "stacked"))
    circ = trotter_circuit(ham, dt, order=2)
    trotter_mat = circ.to_matrix()
    exact_mat = expm(-1j * dt * ham.to_matrix())
    frob_err = np.linalg.norm(trotter_mat - exact_mat, ord="fro")
    assert frob_err < 1e-5, f"Frobenius error {frob_err:.2e}"


def test_fh_trotter_vs_propagator_interleaved():
    num_fermionic_sites, dt = 3, 0.01
    fh = FermiHubbard1D(num_fermionic_sites=num_fermionic_sites, t=1.0, U=2.0, mu=0.0)
    ham = fh.encode(_fh_encoding(num_fermionic_sites, "interleaved"))
    circ = trotter_circuit(ham, dt, order=2)
    trotter_mat = circ.to_matrix()
    exact_mat = expm(-1j * dt * ham.to_matrix())
    frob_err = np.linalg.norm(trotter_mat - exact_mat, ord="fro")
    assert frob_err < 1e-5, f"Frobenius error {frob_err:.2e}"


# ---------------------------------------------------------------------------
# Periodic boundary conditions
# ---------------------------------------------------------------------------


def test_fh_pbc_l1_raises():
    """periodic=True with num_fermionic_sites=1 must raise ValueError."""
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
        fh = FermiHubbard1D(3, t=t, U=U, mu=mu, periodic=True)
        H = fh.encode(_fh_encoding(3, ordering)).to_matrix()
        assert np.allclose(H, H.conj().T, atol=1e-12), (
            f"{ordering} PBC Hamiltonian not Hermitian"
        )


def test_fh_pbc_spectrum_ordering_agreement():
    """PBC stacked and interleaved must give the same spectrum."""
    for num_fermionic_sites in [3, 4]:
        fh = FermiHubbard1D(num_fermionic_sites, t=1.0, U=2.0, mu=0.5, periodic=True)
        E_s = np.linalg.eigvalsh(
            fh.encode(_fh_encoding(num_fermionic_sites, "stacked")).to_matrix()
        )[0]
        E_i = np.linalg.eigvalsh(
            fh.encode(_fh_encoding(num_fermionic_sites, "interleaved")).to_matrix()
        )[0]
        assert abs(E_s - E_i) < 1e-10, (
            f"num_fermionic_sites={num_fermionic_sites}: stacked E0={E_s}, interleaved E0={E_i}"
        )


def test_fh_pbc_vs_obc_spectrum_differs():
    """PBC and OBC give different ground-state energies for L=3."""
    for ordering in ["stacked", "interleaved"]:
        fh_obc = FermiHubbard1D(3, t=1.0, U=1.0, periodic=False)
        fh_pbc = FermiHubbard1D(3, t=1.0, U=1.0, periodic=True)
        encoding = _fh_encoding(3, ordering)
        E_obc = np.linalg.eigvalsh(fh_obc.encode(encoding).to_matrix())[0]
        E_pbc = np.linalg.eigvalsh(fh_pbc.encode(encoding).to_matrix())[0]
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
        fh = FermiHubbard1D(L, t=t, U=0.0, mu=0.0, periodic=True)
        H = fh.encode(_fh_encoding(L, ordering)).to_matrix()
        E0 = np.linalg.eigvalsh(H)[0]
        assert abs(E0 - expected) < 1e-8, (
            f"{ordering}: got E0={E0:.8f}, expected {expected:.8f}"
        )


def test_fh_trotter_error_scaling_order1():
    """Log-log slope of Frobenius error vs dt should be ~2 for order-1 Trotter."""
    num_fermionic_sites = 3
    dt_values = [0.1, 0.05, 0.02, 0.01]
    for ordering in ["stacked", "interleaved"]:
        fh = FermiHubbard1D(
            num_fermionic_sites=num_fermionic_sites, t=1.0, U=2.0, mu=0.0
        )
        ham = fh.encode(_fh_encoding(num_fermionic_sites, ordering))
        errors = []
        for dt in dt_values:
            circ = trotter_circuit(ham, dt, order=1)
            errors.append(
                np.linalg.norm(
                    circ.to_matrix() - expm(-1j * dt * ham.to_matrix()), ord="fro"
                )
            )
        slope = np.polyfit(np.log(dt_values), np.log(errors), 1)[0]
        assert 1.8 < slope < 2.2, (
            f"ordering={ordering}: order-1 slope={slope:.3f} (expected ~2)"
        )


# ---------------------------------------------------------------------------
# interleaved_mode_order annotation
# ---------------------------------------------------------------------------


def test_interleaved_mode_order_annotate_false_is_unchanged():
    assert interleaved_mode_order(2, annotate=False) == interleaved_mode_order(2)


def test_interleaved_mode_order_annotate_returns_order_and_string():
    order, annotation = interleaved_mode_order(2, annotate=True)
    assert order == (0, 2, 1, 3)
    assert annotation == "q0=↑0 q1=↓0 q2=↑1 q3=↓1"
