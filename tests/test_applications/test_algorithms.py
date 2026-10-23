"""Numerical algorithms and the independent references used to verify them."""

from itertools import combinations
from math import comb
import numpy as np
import pytest
import stim
from scipy.linalg import expm
from qaravan.applications.algorithms import (
    constrained_spectrum,
    fermionic_ground_state,
    fermionic_spectrum,
    ground_state,
    krylov_evolve,
    spectrum,
)
from qaravan.backends.statevector import Statevector
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
from qaravan.core.observables import (
    HamiltonianTerm,
    PauliString,
    PauliSum,
    clifford_conjugate,
)

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


def _krylov_generator() -> PauliSum:
    return PauliSum(
        [
            PauliString("III", 0.25),
            PauliString("XII", 0.7),
            PauliString("IYI", -0.4),
            PauliString("IZZ", 1.1),
        ]
    )


# ---------------------------------------------------------------------------
# Krylov time evolution
# ---------------------------------------------------------------------------


def test_krylov_evolve_matches_dense_exponential():
    generator = _krylov_generator()
    initial_state = Statevector(3, random_seed=7)
    time = 0.43

    expected = expm(-1j * time * generator.to_matrix()) @ initial_state.to_array()
    actual = krylov_evolve(generator, initial_state, time)

    np.testing.assert_allclose(actual.to_array(), expected, atol=1e-12)


def test_krylov_evolve_never_constructs_the_generator_matrix(monkeypatch):
    generator = _krylov_generator()
    initial_state = Statevector(3, random_seed=11)
    time = -0.37
    expected = expm(-1j * time * generator.to_matrix()) @ initial_state.to_array()

    def fail_dense_path():
        raise AssertionError("krylov_evolve constructed a dense operator")

    monkeypatch.setattr(generator, "to_matrix", fail_dense_path)
    actual = krylov_evolve(generator, initial_state, time)

    np.testing.assert_allclose(actual.to_array(), expected, atol=1e-12)


def test_krylov_evolve_forward_then_backward_recovers_the_state():
    generator = _krylov_generator()
    initial_state = Statevector(3, random_seed=5)

    forward = krylov_evolve(generator, initial_state, 0.91)
    recovered = krylov_evolve(generator, forward, -0.91)

    assert recovered.overlap(initial_state) == pytest.approx(1.0, abs=1e-12)
    assert recovered.norm() == pytest.approx(1.0, abs=1e-12)


def test_krylov_evolve_identity_is_global_phase():
    coefficient = -0.73
    time = 1.27
    generator = PauliSum([PauliString("III", coefficient)])
    initial_state = Statevector(3, random_seed=3)

    evolved = krylov_evolve(generator, initial_state, time)

    expected = np.exp(-1j * time * coefficient) * initial_state.to_array()
    np.testing.assert_allclose(evolved.to_array(), expected, atol=1e-12)


def test_krylov_evolve_does_not_mutate_initial_state():
    generator = _krylov_generator()
    initial_state = Statevector(3, random_seed=13)
    original = initial_state.to_array().copy()

    krylov_evolve(generator, initial_state, 0.2)

    np.testing.assert_array_equal(initial_state.to_array(), original)


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


# === constrained_spectrum ===
#
# References are independent of the method: the JW popcount-sector spectrum, a dense projection
# P H P built from the constraint matrices, and the free-fermion two-particle spectrum.


def _jw_hubbard(num_fermionic_sites: int = 3) -> PauliSum:
    return (
        FermiHubbard1D(num_fermionic_sites=num_fermionic_sites, t=1.0, U=2.0, mu=0.3)
        .encode(FermionicEncoding("jordan_wigner", num_modes=2 * num_fermionic_sites))
        .as_observable()
    )


def _z(qubit: int, num_qubits: int, sign: int = 1) -> PauliString:
    return PauliString("I" * qubit + "Z" + "I" * (num_qubits - qubit - 1), sign)


def _pad(pauli_sum: PauliSum, num_extra: int) -> PauliSum:
    """Append untouched qubits to every term."""
    return PauliSum(
        [PauliString(t.string + "I" * num_extra, t.coeff) for t in pauli_sum.terms]
    )


def _disguise(pauli_sum: PauliSum, stabilizers, occupations, clifford):
    """Conjugate the whole problem by one Clifford: every Pauli becomes a dense mixed string."""
    conjugate = lambda pauli: clifford_conjugate(PauliSum([pauli]), clifford).terms[0]  # noqa: E731
    return (
        clifford_conjugate(pauli_sum, clifford),
        [conjugate(s) for s in stabilizers],
        [conjugate(o) for o in occupations],
    )


def _dense_constrained_spectrum(
    pauli_sum, stabilizers, occupations, num_particles
) -> np.ndarray:
    """Eigenvalues of P H P on range(P), all matrices dense: P = prod (1 + S)/2 times the count projector."""
    dim = 2**pauli_sum.num_sites
    projector = np.eye(dim, dtype=complex)
    for stabilizer in stabilizers:
        projector = projector @ (np.eye(dim) + stabilizer.to_matrix()) / 2
    occupied = [(np.eye(dim) - o.to_matrix()) / 2 for o in occupations]
    count = sum(occupied)
    weights, vectors = np.linalg.eigh(count)
    number_projector = (
        vectors[:, np.isclose(weights, num_particles)]
        @ vectors[:, np.isclose(weights, num_particles)].conj().T
    )
    projector = projector @ number_projector
    weights, vectors = np.linalg.eigh((projector + projector.conj().T) / 2)
    basis = vectors[:, weights > 0.5]
    return np.linalg.eigvalsh(basis.conj().T @ pauli_sum.to_matrix() @ basis)


@pytest.mark.parametrize("num_particles", range(7))
def test_constrained_spectrum_without_stabilizers_is_the_popcount_spectrum(
    num_particles,
):
    """Catches a wrong bit order or sector listing on the plain JW case."""
    hamiltonian = _jw_hubbard()
    occupations = [_z(q, 6) for q in range(6)]
    np.testing.assert_allclose(
        constrained_spectrum(hamiltonian, [], occupations, num_particles),
        spectrum(hamiltonian, fixed_particle_number=num_particles),
        atol=1e-10,
    )


@pytest.mark.parametrize("num_particles", [1, 2, 3])
def test_disguised_jordan_wigner_with_ancillas_recovers_the_jw_spectrum(num_particles):
    """JW Hubbard plus 2 ancillas pinned by Z stabilizers, all scrambled by a random Clifford. Catches sign or
    phase errors in rotating and tapering, which the scramble spreads over every term."""
    hamiltonian = _pad(_jw_hubbard(), 2)
    stabilizers = [_z(6, 8), _z(7, 8, -1)]
    occupations = [_z(q, 8) for q in range(6)]
    disguised = _disguise(hamiltonian, stabilizers, occupations, stim.Tableau.random(8))
    np.testing.assert_allclose(
        constrained_spectrum(*disguised, num_particles),
        spectrum(_jw_hubbard(), fixed_particle_number=num_particles),
        atol=1e-10,
    )


def test_stabilizer_signs_select_different_spectra():
    """A 0.4 Z_anc term reads +-0.4 depending on the ancilla's stabilizer sign: the two signs must give the
    dense reference's two spectra, 0.8 apart. Catches a method blind to stabilizer signs."""
    hamiltonian = _pad(_jw_hubbard(2), 1) + PauliString("IIIIZ", 0.4)
    occupations = [_z(q, 5) for q in range(4)]
    clifford = stim.Tableau.random(5)
    results = []
    for sign in (1, -1):
        problem = _disguise(hamiltonian, [_z(4, 5, sign)], occupations, clifford)
        expected = _dense_constrained_spectrum(*problem, 2)
        np.testing.assert_allclose(
            constrained_spectrum(*problem, 2), expected, atol=1e-10
        )
        results.append(expected)
    np.testing.assert_allclose(results[0] - results[1], 0.8, atol=1e-10)


def test_even_parity_code_space_keeps_even_particle_numbers_only():
    """Total parity as a stabilizer makes the occupation operators dependent on it. Even k must give the JW
    spectrum, odd k nothing. Catches the dependency constraints being ignored or mis-signed."""
    hamiltonian = _jw_hubbard(2)
    parity = PauliString("ZZZZ")
    occupations = [_z(q, 4) for q in range(4)]
    disguised = _disguise(hamiltonian, [parity], occupations, stim.Tableau.random(4))
    for k in range(5):
        got = constrained_spectrum(*disguised, k)
        expected = (
            spectrum(hamiltonian, fixed_particle_number=k)
            if k % 2 == 0
            else np.array([])
        )
        np.testing.assert_allclose(got, expected, atol=1e-10)


def test_unconstrained_qubits_multiply_every_eigenvalue():
    """2 qubits pinned by nothing: each JW sector eigenvalue appears 4 times."""
    occupations = [_z(q, 6) for q in range(4)]
    disguised = _disguise(
        _pad(_jw_hubbard(2), 2), [], occupations, stim.Tableau.random(6)
    )
    expected = np.repeat(spectrum(_jw_hubbard(2), fixed_particle_number=2), 4)
    np.testing.assert_allclose(
        constrained_spectrum(*disguised, 2), expected, atol=1e-10
    )


def test_dependent_stabilizers_are_ignored_and_inconsistent_ones_raise():
    hamiltonian = _pad(_jw_hubbard(2), 1)
    occupations = [_z(q, 5) for q in range(4)]
    duplicated = constrained_spectrum(hamiltonian, [_z(4, 5), _z(4, 5)], occupations, 2)
    np.testing.assert_allclose(
        duplicated, spectrum(_jw_hubbard(2), fixed_particle_number=2), atol=1e-10
    )
    with pytest.raises(ValueError, match="no state satisfies"):
        constrained_spectrum(hamiltonian, [_z(4, 5), _z(4, 5, -1)], occupations, 2)


def test_hamiltonian_not_commuting_with_a_stabilizer_raises():
    hamiltonian = _pad(_jw_hubbard(2), 1) + PauliString("IIIIX", 0.1)
    with pytest.raises(ValueError, match="does not commute"):
        constrained_spectrum(hamiltonian, [_z(4, 5)], [_z(q, 5) for q in range(4)], 2)


def test_seventy_qubit_free_fermion_chain_at_two_particles():
    """66-mode hopping chain plus 4 stabilized ancillas, disguised (beyond int64 basis indices after tapering).
    k = 2 eigenvalues must be the sums of distinct single-particle energies. Catches anything exponential and
    the Python-int sector path."""
    num_modes, num_qubits = 66, 70
    rng = np.random.default_rng(0)
    potentials = rng.normal(size=num_modes)
    single_particle = (
        np.diag(potentials) - np.eye(num_modes, k=1) - np.eye(num_modes, k=-1)
    )
    terms = []
    for j in range(num_modes - 1):
        for letter in "XY":
            string = "I" * j + letter * 2 + "I" * (num_qubits - j - 2)
            terms.append(PauliString(string, -0.5))
    for j, mu in enumerate(potentials):
        terms += [
            PauliString("I" * num_qubits, mu / 2),
            PauliString(_z(j, num_qubits).string, -mu / 2),
        ]
    stabilizers = [_z(q, num_qubits) for q in range(num_modes, num_qubits)]
    occupations = [_z(q, num_qubits) for q in range(num_modes)]
    disguised = _disguise(
        PauliSum(terms), stabilizers, occupations, stim.Tableau.random(num_qubits)
    )

    energies = np.linalg.eigvalsh(single_particle)
    expected = np.sort(
        [energies[a] + energies[b] for a, b in combinations(range(num_modes), 2)]
    )
    np.testing.assert_allclose(constrained_spectrum(*disguised, 2), expected, atol=1e-9)
