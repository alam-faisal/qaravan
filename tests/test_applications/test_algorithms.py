"""Eigensolvers in applications/algorithms.py, plus the spin-register derivation they use."""

from math import comb
import numpy as np
import pytest
from qaravan.applications.algorithms import (
    fermionic_ground_state,
    fermionic_spectrum,
    ground_state,
    spectrum,
)
from qaravan.core.fermions import (
    FermionicEncoding,
    number,
)
from qaravan.core.hamiltonians import (
    TFI,
    FermiHubbard1D,
    FermionicHamiltonian,
    Hamiltonian,
    Heisenberg1D,
    bath_interleaved_mode_order,
    interleaved_mode_order,
)
from qaravan.core.observables import HamiltonianTerm, PauliString

# stacked and interleaved up/down qubit registers, written out by hand rather than
# taken from spin_qubits(), so the sector tests below stay independent of it
STACKED_QUBITS = ([0, 1, 2], [3, 4, 5])
INTERLEAVED_QUBITS = ([0, 2, 4], [1, 3, 5])


def _fh_encoding(num_fermionic_sites: int, ordering: str) -> FermionicEncoding:
    mode_order = (
        None if ordering == "stacked" else interleaved_mode_order(num_fermionic_sites)
    )
    return FermionicEncoding(
        "jordan_wigner", num_modes=2 * num_fermionic_sites, mode_order=mode_order
    )


def _spin_counts(index, num_sites, up_qubits, down_qubits) -> tuple[int, int]:
    """(n_up, n_down) of a basis index, read off its bitstring; bits[q] is qubit q."""
    bits = format(index, f"0{num_sites}b")
    return (
        sum(bits[qubit] == "1" for qubit in up_qubits),
        sum(bits[qubit] == "1" for qubit in down_qubits),
    )


def _zeeman_fh(num_fermionic_sites: int, zeeman: float) -> FermionicHamiltonian:
    """FermiHubbard1D plus a field on the up modes only, which splits (n_up, n_down) from
    (n_down, n_up) -- the plain model is spin-flip symmetric and cannot tell them apart."""
    fh = FermiHubbard1D(num_fermionic_sites=num_fermionic_sites, t=1.0, U=2.0)
    terms = list(fh.terms) + [
        zeeman * number(mode) for mode in range(num_fermionic_sites)
    ]
    return FermionicHamiltonian(terms, 2 * num_fermionic_sites)


def _projected_ground_energy(encoded, spin_numbers, up_qubits, down_qubits) -> float:
    """Lowest eigenvalue of the (n_up, n_down) block, projected using bitstrings alone."""
    num_sites = encoded.num_sites
    keep = [
        index
        for index in range(2**num_sites)
        if _spin_counts(index, num_sites, up_qubits, down_qubits) == spin_numbers
    ]
    return float(np.linalg.eigvalsh(encoded.to_matrix()[np.ix_(keep, keep)])[0])


# ---------------------------------------------------------------------------
# FermionicEncoding.spin_qubits
# ---------------------------------------------------------------------------


def test_spin_qubits_stacked_layout():
    # mode_order=None is the identity, so up modes 0,1,2 stay on qubits 0,1,2
    assert _fh_encoding(3, "stacked").spin_qubits() == STACKED_QUBITS


def test_spin_qubits_interleaved_layout():
    # interleaved_mode_order puts up_j on qubit 2j and down_j on qubit 2j+1
    assert _fh_encoding(3, "interleaved").spin_qubits() == INTERLEAVED_QUBITS


def test_spin_qubits_aim_bath_interleaved():
    # bath_interleaved_mode_order(2) = (1, 4, 2, 5, 0, 3); modes 0,1,2 are up, 3,4,5 down
    encoding = FermionicEncoding(
        "jordan_wigner", num_modes=6, mode_order=bath_interleaved_mode_order(2)
    )
    assert encoding.spin_qubits() == ([4, 0, 2], [5, 1, 3])


def test_spin_qubits_is_a_partition_of_all_qubits():
    for ordering in ["stacked", "interleaved"]:
        up, down = _fh_encoding(4, ordering).spin_qubits()
        assert sorted(up + down) == list(range(8)), ordering


def test_spin_qubits_raises_on_odd_mode_count():
    encoding = FermionicEncoding("jordan_wigner", num_modes=3, mode_order=(2, 0, 1))
    with pytest.raises(ValueError, match="even mode count"):
        encoding.spin_qubits()


# ---------------------------------------------------------------------------
# Spin sectors -- the bit-ordering regression
#
# Qubit q is bit (num_sites-1-q) of the basis index. Getting that backwards maps every
# up register in this repo exactly onto its down register (all three layouts are mirror
# symmetric under q -> n-1-q), so it is a clean up<->down swap. That makes the two
# obvious tests useless here: stacked-vs-interleaved agreement (both swap identically)
# and (2,1)-vs-(1,2) energies (spin-flip degenerate). The two below do catch it.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ordering,registers",
    [("stacked", STACKED_QUBITS), ("interleaved", INTERLEAVED_QUBITS)],
)
def test_spin_sector_state_has_no_amplitude_outside_the_sector(ordering, registers):
    """Amplitudes are checked against a bitstring-derived projector, not the solver's own."""
    fh = FermiHubbard1D(num_fermionic_sites=3, t=1.0, U=2.0)
    encoding = _fh_encoding(3, ordering)
    up_qubits, down_qubits = registers

    sv = fermionic_ground_state(fh, encoding, fixed_spin_numbers=(2, 1))
    amplitudes = sv.to_array()

    in_sector = [
        index
        for index in range(2**6)
        if _spin_counts(index, 6, up_qubits, down_qubits) == (2, 1)
    ]
    outside = np.delete(amplitudes, in_sector)
    assert np.max(np.abs(outside)) < 1e-12
    assert np.linalg.norm(amplitudes[in_sector]) == pytest.approx(1.0, abs=1e-10)


def test_spin_sector_energies_split_under_an_up_only_field():
    """A field on the up modes breaks the (2,1)/(1,2) degeneracy, so a swapped register
    assignment shows up as the wrong energy."""
    ham = _zeeman_fh(2, zeeman=0.7)
    encoding = FermionicEncoding("jordan_wigner", num_modes=4)
    encoded = ham.encode(encoding)
    up_qubits, down_qubits = [0, 1], [2, 3]

    e_up_heavy = fermionic_spectrum(
        ham, encoding, num_eigenvalues=1, fixed_spin_numbers=(2, 1)
    )[0]
    e_down_heavy = fermionic_spectrum(
        ham, encoding, num_eigenvalues=1, fixed_spin_numbers=(1, 2)
    )[0]
    assert abs(e_up_heavy - e_down_heavy) > 1e-6, "field failed to split the sectors"

    for spin_numbers, energy in [((2, 1), e_up_heavy), ((1, 2), e_down_heavy)]:
        expected = _projected_ground_energy(
            encoded, spin_numbers, up_qubits, down_qubits
        )
        assert energy == pytest.approx(expected, abs=1e-10), spin_numbers


def test_symmetric_spin_sector_energy_is_layout_independent():
    """(1,1) is what the existing notebook relies on; it must not move."""
    fh = FermiHubbard1D(num_fermionic_sites=2, t=1.0, U=2.0)
    energies = [
        fermionic_spectrum(
            fh, _fh_encoding(2, ordering), num_eigenvalues=1, fixed_spin_numbers=(1, 1)
        )[0]
        for ordering in ["stacked", "interleaved"]
    ]
    assert energies[0] == pytest.approx(energies[1], abs=1e-10)


def test_both_sector_arguments_agree():
    fh = FermiHubbard1D(num_fermionic_sites=2, t=1.0, U=2.0)
    encoding = _fh_encoding(2, "stacked")
    both = fermionic_spectrum(
        fh,
        encoding,
        num_eigenvalues=1,
        fixed_particle_number=3,
        fixed_spin_numbers=(2, 1),
    )
    spin_only = fermionic_spectrum(
        fh, encoding, num_eigenvalues=1, fixed_spin_numbers=(2, 1)
    )
    np.testing.assert_allclose(both, spin_only, atol=1e-12)


def test_inconsistent_sector_arguments_raise():
    fh = FermiHubbard1D(num_fermionic_sites=2, t=1.0, U=2.0)
    with pytest.raises(ValueError, match="sums to 3"):
        fermionic_ground_state(
            fh,
            _fh_encoding(2, "stacked"),
            fixed_particle_number=2,
            fixed_spin_numbers=(2, 1),
        )


def test_fixed_particle_number_and_sector_are_exclusive():
    tfi = TFI(3)
    with pytest.raises(ValueError, match="at most one"):
        ground_state(tfi, fixed_particle_number=1, sector=np.array([0, 1]))


# ---------------------------------------------------------------------------
# Method agreement and spectrum semantics
# ---------------------------------------------------------------------------


def test_naive_and_naive_lanczos_agree_on_ground_energy():
    for n in [3, 4, 5]:
        tfi = TFI(n, J=1.0, h=0.5)
        energies = [
            ground_state(tfi, method=method).expectation(tfi.as_observable()).real
            for method in ["naive", "naive_lanczos"]
        ]
        assert energies[0] == pytest.approx(energies[1], abs=1e-9), n


def test_naive_and_naive_lanczos_agree_on_low_lying_spectrum():
    tfi = TFI(5, J=1.0, h=0.5)
    naive = spectrum(tfi, method="naive", num_eigenvalues=4)
    lanczos = spectrum(tfi, method="naive_lanczos", num_eigenvalues=4)
    np.testing.assert_allclose(naive, lanczos, atol=1e-9)


def test_spectrum_default_is_the_full_ascending_spectrum():
    tfi = TFI(4, J=1.0, h=0.5)
    np.testing.assert_allclose(
        spectrum(tfi), np.linalg.eigvalsh(tfi.to_matrix()), atol=1e-10
    )


def test_spectrum_num_eigenvalues_slices_the_bottom():
    tfi = TFI(4, J=1.0, h=0.5)
    full = spectrum(tfi)
    np.testing.assert_allclose(spectrum(tfi, num_eigenvalues=3), full[:3], atol=1e-10)


def test_sector_restricted_spectrum_has_binomial_length():
    fh = FermiHubbard1D(num_fermionic_sites=3, t=1.0, U=2.0)
    evals = fermionic_spectrum(fh, _fh_encoding(3, "stacked"), fixed_particle_number=3)
    assert len(evals) == comb(6, 3)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        ground_state(TFI(3), method="bogus")


@pytest.mark.parametrize("method", ["naive_lanczos", "lanczos"])
def test_lanczos_methods_need_num_eigenvalues(method):
    with pytest.raises(ValueError, match="needs num_eigenvalues"):
        spectrum(TFI(4), method=method)


def test_naive_lanczos_raises_for_a_tiny_matrix():
    tiny = Hamiltonian([HamiltonianTerm([0], PauliString("Z"))], 1)
    with pytest.raises(ValueError, match="naive"):
        ground_state(tiny, method="naive_lanczos")


# ---------------------------------------------------------------------------
# Matrix-free Lanczos
# ---------------------------------------------------------------------------


def test_matrix_free_lanczos_agrees_on_the_low_lying_spectrum():
    tfi = TFI(6, J=1.0, h=0.7)
    np.testing.assert_allclose(
        spectrum(tfi, method="lanczos", num_eigenvalues=4),
        spectrum(tfi, method="naive", num_eigenvalues=4),
        atol=1e-10,
    )


def test_matrix_free_lanczos_ground_state_matches_the_dense_one():
    """Overlap rather than elementwise: eigenvectors are defined only up to a phase."""
    tfi = TFI(6, J=1.0, h=0.7)
    dense = ground_state(tfi, method="naive").to_array()
    free = ground_state(tfi, method="lanczos").to_array()
    assert abs(np.vdot(dense, free)) == pytest.approx(1.0, abs=1e-10)


def test_matrix_free_lanczos_in_a_popcount_sector():
    """The Krylov vectors live in the sector, so this also pins that _embed scatters the
    result back to the right 2^n positions."""
    heisenberg = Heisenberg1D(6, J=1.0, h=0.3)
    np.testing.assert_allclose(
        spectrum(heisenberg, "lanczos", 3, fixed_particle_number=3),
        spectrum(heisenberg, "naive", 3, fixed_particle_number=3),
        atol=1e-10,
    )
    dense = ground_state(heisenberg, "naive", fixed_particle_number=3).to_array()
    free = ground_state(heisenberg, "lanczos", fixed_particle_number=3).to_array()
    assert abs(np.vdot(dense, free)) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("mode_order_name", ["stacked", "interleaved"])
def test_matrix_free_lanczos_in_a_spin_sector(mode_order_name):
    """Catches a bit-convention mismatch between _fermionic_sector and PauliString.bitmasks --
    the two places that independently encode 'qubit q is bit n-1-q'."""
    fh = FermiHubbard1D(num_fermionic_sites=3, t=1.0, U=2.0)
    encoding = _fh_encoding(3, mode_order_name)
    kwargs = dict(fixed_spin_numbers=(2, 1), num_eigenvalues=2)
    np.testing.assert_allclose(
        fermionic_spectrum(fh, encoding, method="lanczos", **kwargs),
        fermionic_spectrum(fh, encoding, method="naive", **kwargs),
        atol=1e-10,
    )
