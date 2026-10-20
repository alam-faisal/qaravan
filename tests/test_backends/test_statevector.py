"""Tests for backends/statevector.py — Statevector and StatevectorSimulator."""

import numpy as np
import pytest
from qaravan.backends.statevector import (
    Statevector,
    StatevectorSimulator,
    apply_pauli_sum,
)
from qaravan.core.base import IncompatibleNoiseError, IncompatibleStateError, NoiseModel
from qaravan.core.circuit import Circuit
from qaravan.core.gates import CNOT, H
from qaravan.core.noise import PauliNoise
from qaravan.core.observables import LocalOp, Magnetization, PauliString, PauliSum


class _MinimalNoise(NoiseModel):
    """Minimal concrete NoiseModel without sampled-error support."""

    @property
    def gate_dependent(self) -> bool:
        return False

    def get_kraus(self, *args, **kwargs):
        return [np.eye(2, dtype=complex)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_H_MAT = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_Z_MAT = np.array([[1, 0], [0, -1]], dtype=complex)


def _bell_state() -> Statevector:
    """(|00⟩ + |11⟩)/√2 built via simulator."""
    circ = Circuit([H(0), CNOT([0, 1])], num_sites=2)
    sv0 = Statevector(2)
    return StatevectorSimulator(circ, sv0).run()


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_statevector_init_allzero():
    sv = Statevector(3)
    assert sv._tensor[0, 0, 0] == pytest.approx(1.0 + 0j)
    assert np.sum(np.abs(sv._tensor) ** 2) == pytest.approx(1.0)


def test_statevector_init_bitstring():
    sv = Statevector(bitstring="01")
    # big-endian: "01" → index int("01", 2) = 1 → _tensor[0, 1] = 1
    assert sv._tensor[0, 1] == pytest.approx(1.0 + 0j)
    assert sv._tensor[1, 0] == pytest.approx(0.0 + 0j)
    assert sv.num_sites == 2


def test_statevector_init_random_seed():
    sv1 = Statevector(4, random_seed=42)
    sv2 = Statevector(4, random_seed=42)
    assert np.allclose(sv1._tensor, sv2._tensor)


def test_statevector_init_different_seeds_differ():
    sv1 = Statevector(3, random_seed=0)
    sv2 = Statevector(3, random_seed=1)
    assert not np.allclose(sv1._tensor, sv2._tensor)


def test_statevector_init_array():
    arr = np.array([1, 0, 0, 0], dtype=complex)
    sv = Statevector(array=arr)
    assert sv.num_sites == 2
    assert sv._tensor[0, 0] == pytest.approx(1.0 + 0j)


def test_statevector_init_array_unnormalized_raises():
    arr = np.array([1, 1, 0, 0], dtype=complex)  # not normalized
    with pytest.raises(ValueError, match="normalized"):
        Statevector(array=arr)


def test_statevector_init_ambiguous_raises():
    with pytest.raises(ValueError):
        Statevector(2, bitstring="00")


def test_statevector_init_no_args_raises():
    with pytest.raises(ValueError):
        Statevector()


# ---------------------------------------------------------------------------
# Basic attributes and helpers
# ---------------------------------------------------------------------------


def test_statevector_norm():
    sv = Statevector(4, random_seed=7)
    assert sv.norm() == pytest.approx(1.0)


def test_statevector_to_array_shape():
    sv = Statevector(3)
    arr = sv.to_array()
    assert arr.shape == (8,)


def test_statevector_to_array_correct():
    sv = Statevector(bitstring="10")
    arr = sv.to_array()
    expected = np.zeros(4, dtype=complex)
    expected[2] = 1.0  # "10" → index 2
    assert np.allclose(arr, expected)


# ---------------------------------------------------------------------------
# Simulator: GHZ and noise
# ---------------------------------------------------------------------------


def test_statevector_sim_ghz():
    """H(0) + CNOT(0,1) on |00⟩ → (|00⟩ + |11⟩)/√2."""
    circ = Circuit([H(0), CNOT([0, 1])], num_sites=2)
    sv0 = Statevector(2)
    sv = StatevectorSimulator(circ, sv0).run()
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(sv.to_array(), expected)


def test_statevector_sim_ghz_3qubit():
    """H(0) + CNOT(0,1) + CNOT(0,2) on |000⟩ → (|000⟩ + |111⟩)/√2."""
    circ = Circuit([H(0), CNOT([0, 1]), CNOT([0, 2])], num_sites=3)
    sv0 = Statevector(3)
    sv = StatevectorSimulator(circ, sv0).run()
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[7] = 1 / np.sqrt(2)
    assert np.allclose(sv.to_array(), expected)


def test_statevector_sim_rejects_noise_without_a_sampler():
    """Kraus-only noise fails at construction, before any evolution."""
    circ = Circuit([H(0)], num_sites=1)
    with pytest.raises(IncompatibleNoiseError, match="sample_error"):
        StatevectorSimulator(circ, Statevector(1), noise_model=_MinimalNoise())


def test_statevector_sim_rejects_unknown_noise_placement():
    """A typo'd placement must raise, not silently fall back to per-layer noise."""
    noise = PauliNoise(["I"], [1.0], placement="circiut")
    with pytest.raises(ValueError, match="circiut"):
        Statevector(1).apply(Circuit([H(0)], num_sites=1), noise_model=noise)


def test_statevector_apply_supports_one_seeded_noisy_path():
    circuit = Circuit([H(0)], num_sites=1)
    noise = PauliNoise(["Z"], [1.0], placement="circuit")

    final = Statevector(bitstring="0").apply(
        circuit,
        noise_model=noise,
        noise_rng=np.random.default_rng(3),
    )

    np.testing.assert_allclose(final.to_array(), np.array([1, -1]) / np.sqrt(2))


def test_statevector_noise_insertion_does_not_mutate_prepared_circuit():
    circuit = Circuit([H(0), H(1), CNOT([0, 1])], num_sites=2)
    noise = PauliNoise(["XI"], [1.0], placement="layer")
    simulator = StatevectorSimulator(
        circuit,
        Statevector(bitstring="00"),
        noise_model=noise,
        noise_rng=np.random.default_rng(0),
    )
    prepared = simulator._prepare_circuit()
    original_layers = [list(layer) for layer in prepared.layers]

    first = simulator._insert_noise(prepared, noise)
    second = simulator._insert_noise(prepared, noise)

    assert prepared.layers == original_layers
    assert first is not prepared
    assert second is not prepared
    assert len(first.layers) == 2 * len(original_layers)
    assert len(second.layers) == 2 * len(original_layers)


def test_statevector_sim_rejects_wrong_state():
    from qaravan.core.base import State

    class OtherState(State):
        @property
        def default_simulator(self):
            return StatevectorSimulator

        def expectation(self, o):
            return 0.0

        def sample(self, s):
            return np.array([])

        def measure_and_collapse(self, s):
            return (self, "0")

        def overlap(self, o):
            return 0.0

        def __repr__(self):
            return "OtherState()"

    circ = Circuit([H(0)], num_sites=1)
    with pytest.raises(IncompatibleStateError):
        StatevectorSimulator(circ, OtherState())


# ---------------------------------------------------------------------------
# Expectation values
# ---------------------------------------------------------------------------


def test_statevector_expectation_pauli_z_on_zero():
    """⟨Z⟩ on |0⟩ = 1."""
    sv = Statevector(1)
    result = sv.expectation(PauliString("Z"))
    assert np.isclose(result, 1.0)


def test_statevector_expectation_pauli_z_on_one():
    """⟨Z⟩ on |1⟩ = -1."""
    sv = Statevector(bitstring="1")
    result = sv.expectation(PauliString("Z"))
    assert np.isclose(result, -1.0)


def test_statevector_expectation_pauli_x_on_plus():
    """⟨X⟩ on |+⟩ = 1."""
    arr = np.array([1, 1], dtype=complex) / np.sqrt(2)
    sv = Statevector(array=arr)
    result = sv.expectation(PauliString("X"))
    assert np.isclose(result, 1.0)


def test_statevector_expectation_pauli_zz_on_bell():
    """⟨ZZ⟩ on (|00⟩+|11⟩)/√2 = 1."""
    sv = _bell_state()
    result = sv.expectation(PauliString("ZZ"))
    assert np.isclose(result, 1.0)


def test_statevector_expectation_pauli_xx_on_bell():
    """⟨XX⟩ on (|00⟩+|11⟩)/√2 = 1."""
    sv = _bell_state()
    result = sv.expectation(PauliString("XX"))
    assert np.isclose(result, 1.0)


def test_statevector_expectation_pauli_coeff():
    """⟨0.5*Z⟩ on |0⟩ = 0.5."""
    sv = Statevector(1)
    result = sv.expectation(PauliString("Z", coeff=0.5))
    assert np.isclose(result, 0.5)


def test_statevector_expectation_pauli_sum():
    """PauliSum Z+Z on |0⟩ = 2."""
    sv = Statevector(1)
    obs = PauliString("Z") + PauliString("Z")
    result = sv.expectation(obs)
    assert np.isclose(result, 2.0)


def test_statevector_expectation_local_op():
    """LocalOp(Z, [0]) on |0⟩ = 1."""
    sv = Statevector(1)
    result = sv.expectation(LocalOp(_Z_MAT, [0]))
    assert np.isclose(result, 1.0)


def test_statevector_expectation_local_op_two_qubit():
    """LocalOp(Z⊗Z, [0,1]) on Bell = 1."""
    sv = _bell_state()
    ZZ = np.kron(_Z_MAT, _Z_MAT)
    result = sv.expectation(LocalOp(ZZ, [0, 1]))
    assert np.isclose(result, 1.0)


def test_statevector_expectation_magnetization():
    """Magnetization(2, 'Z') on |01⟩ = 0."""
    sv = Statevector(bitstring="01")
    result = sv.expectation(Magnetization(2, "Z"))
    assert np.isclose(result, 0.0)


def test_statevector_expectation_unsupported_raises():
    from qaravan.core.base import Observable

    class WeirdObs(Observable):
        def to_matrix(self):
            return np.eye(2, dtype=complex)

    sv = Statevector(1)
    with pytest.raises(NotImplementedError):
        sv.expectation(WeirdObs("W", [0]))


# ---------------------------------------------------------------------------
# RDM, overlap, project_and_renorm
# ---------------------------------------------------------------------------


def test_statevector_rdm_diagonal():
    """rdm([0]) of |0⟩ = [[1,0],[0,0]]."""
    sv = Statevector(1)
    rdm = sv.rdm([0])
    expected = np.array([[1, 0], [0, 0]], dtype=complex)
    assert np.allclose(rdm, expected)


def test_statevector_rdm_bell():
    """Each qubit's RDM in Bell state is maximally mixed: I/2."""
    sv = _bell_state()
    rdm0 = sv.rdm([0])
    assert np.allclose(rdm0, np.eye(2) / 2, atol=1e-12)


def test_statevector_overlap_self():
    sv = Statevector(3, random_seed=5)
    assert np.isclose(sv.overlap(sv), 1.0)


def test_statevector_overlap_orthogonal():
    sv0 = Statevector(1)
    sv1 = Statevector(bitstring="1")
    assert np.isclose(abs(sv0.overlap(sv1)), 0.0)


# ---------------------------------------------------------------------------
# partial_overlap
# ---------------------------------------------------------------------------


def test_partial_overlap_full_returns_1x1():
    """partial_overlap with skip=[] returns shape (1,1)."""
    sv = Statevector(3, random_seed=7)
    mat = sv.partial_overlap(sv, skip=[])
    assert mat.shape == (1, 1)


def test_partial_overlap_full_equals_overlap():
    """partial_overlap(skip=[])[0,0].conj() == overlap() for two different states.

    partial_overlap[0,0] = ⟨other|self⟩; overlap = ⟨self|other⟩ = conj of that.
    Using two distinct states so the imaginary part is non-trivial.
    """
    sv1 = Statevector(4, random_seed=7)
    sv2 = Statevector(4, random_seed=11)
    mat = sv1.partial_overlap(sv2, skip=[])
    assert mat.shape == (1, 1)
    assert np.isclose(mat[0, 0].conj(), sv1.overlap(sv2))


def test_partial_overlap_equals_rdm_single_site():
    """partial_overlap(sv, sv, skip=[0]) == rdm([0])."""
    sv = Statevector(4, random_seed=13)
    po = sv.partial_overlap(sv, skip=[0])
    assert np.allclose(po, sv.rdm([0]), atol=1e-12)


def test_partial_overlap_equals_rdm_two_sites():
    """partial_overlap(sv, sv, skip=[0, 2]) == rdm([0, 2])."""
    sv = Statevector(4, random_seed=21)
    po = sv.partial_overlap(sv, skip=[0, 2])
    assert np.allclose(po, sv.rdm([0, 2]), atol=1e-12)


def test_partial_overlap_cross_state_shape():
    """partial_overlap of two different states with skip=[1] returns (2,2)."""
    sv1 = Statevector(3, random_seed=1)
    sv2 = Statevector(3, random_seed=2)
    po = sv1.partial_overlap(sv2, skip=[1])
    assert po.shape == (2, 2)


def test_partial_overlap_cross_state_manual():
    """result[i,j] = ⟨sv2[j]|sv1[i]⟩ = sum_{a,b} sv1[a,i,b] * conj(sv2[a,j,b]) for skip=[1]."""
    sv1 = Statevector(3, random_seed=1)
    sv2 = Statevector(3, random_seed=2)
    po = sv1.partial_overlap(sv2, skip=[1])
    # Convention: sv1=self is in ket position (index i), sv2=other is in bra position (index j)
    # result[i,j] = sum_{a,b} sv1._tensor[a,i,b] * conj(sv2._tensor[a,j,b])
    manual = np.einsum("aib,ajb->ij", sv1._tensor, sv2._tensor.conj())
    assert np.allclose(po, manual, atol=1e-12)


def test_statevector_project_and_renorm():
    """Project |+⟩ onto '0' → |0⟩."""
    arr = np.array([1, 1], dtype=complex) / np.sqrt(2)
    sv = Statevector(array=arr)
    sv_projected = sv.project_and_renorm([0], "0")
    expected = np.array([1, 0], dtype=complex)
    assert np.allclose(sv_projected.to_array(), expected)


def test_statevector_project_and_renorm_two_sites():
    """Project Bell on [0,1]='00' → |00⟩."""
    sv = _bell_state()
    sv_projected = sv.project_and_renorm([0, 1], "00")
    expected = np.array([1, 0, 0, 0], dtype=complex)
    assert np.allclose(sv_projected.to_array(), expected)


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


def test_statevector_sample_shape():
    sv = Statevector(3)
    shots = sv.sample(100)
    assert shots.shape == (100, 3)
    assert shots.dtype == np.int8


def test_statevector_sample_all_zeros_state():
    """All-zero state must always sample |000⟩."""
    sv = Statevector(3)
    shots = sv.sample(50)
    assert np.all(shots == 0)


def test_statevector_sample_born_rule():
    """Equal superposition on 1 qubit: ~50% 0, ~50% 1 over many shots."""
    arr = np.array([1, 1], dtype=complex) / np.sqrt(2)
    sv = Statevector(array=arr)
    shots = sv.sample(10000)
    frac_zero = np.mean(shots[:, 0] == 0)
    assert abs(frac_zero - 0.5) < 0.02


# ---------------------------------------------------------------------------
# measure_and_collapse
# ---------------------------------------------------------------------------


def test_statevector_sample_and_collapse():
    """Measure |+⟩ 1000 times: ~50/50 split and collapsed state correct."""
    arr = np.array([1, 1], dtype=complex) / np.sqrt(2)
    sv = Statevector(array=arr)

    counts = {"0": 0, "1": 0}
    for _ in range(1000):
        sv_post, outcome = sv.measure_and_collapse([0])
        counts[outcome] += 1
        if outcome == "0":
            assert np.allclose(sv_post.to_array(), np.array([1, 0], dtype=complex))
        else:
            assert np.allclose(sv_post.to_array(), np.array([0, 1], dtype=complex))

    # Born-rule fractions within 3σ for N=1000
    assert abs(counts["0"] / 1000 - 0.5) < 0.05


def test_statevector_measure_and_collapse_bell():
    """Measuring qubit 0 of Bell state collapses correctly."""
    sv = _bell_state()
    sv_post, outcome = sv.measure_and_collapse([0])
    if outcome == "0":
        expected = np.array([1, 0, 0, 0], dtype=complex)
    else:
        expected = np.array([0, 0, 0, 1], dtype=complex)
    assert np.allclose(sv_post.to_array(), expected)


def test_statevector_measure_and_collapse_returns_tuple():
    sv = Statevector(2)
    result = sv.measure_and_collapse([0])
    assert isinstance(result, tuple) and len(result) == 2
    sv_post, outcome = result
    assert isinstance(sv_post, Statevector)
    assert isinstance(outcome, str)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


def test_statevector_reset_one_to_zero():
    """reset |1⟩ to 0 gives |0⟩."""
    sv = Statevector(bitstring="1")
    sv_reset = sv.reset([0], reset_to=0)
    assert np.allclose(sv_reset.to_array(), np.array([1, 0], dtype=complex))


def test_statevector_reset_zero_is_noop():
    """reset |0⟩ to 0 is a no-op."""
    sv = Statevector(1)
    sv_reset = sv.reset([0], reset_to=0)
    assert np.allclose(sv_reset.to_array(), sv.to_array())


def test_statevector_reset_preserves_num_sites():
    """reset on one qubit of a 2-qubit state keeps num_sites=2."""
    sv = Statevector(2)
    sv_reset = sv.reset([0])
    assert sv_reset.num_sites == 2


# ---------------------------------------------------------------------------
# default_simulator
# ---------------------------------------------------------------------------


def test_statevector_default_simulator():
    sv = Statevector(2)
    assert sv.default_simulator is StatevectorSimulator


# ---------------------------------------------------------------------------
# apply_pauli_sum
# ---------------------------------------------------------------------------


def _mixed_pauli_sum(num_sites: int) -> PauliSum:
    """Complex coefficients and every Pauli character, so a transposed accumulation shows up."""
    return PauliSum(
        [
            PauliString("XYIZII"[:num_sites].ljust(num_sites, "I"), 0.7 - 0.2j),
            PauliString("ZZIIXY"[:num_sites].ljust(num_sites, "I"), -0.4),
            PauliString("YIXIZI"[:num_sites].ljust(num_sites, "I"), 1.1j),
            PauliString("I" * num_sites, 0.25),
        ]
    )


def test_apply_pauli_sum_matches_the_dense_matvec():
    """Catches a transposed accumulation, which is invisible for a Hermitian real-coefficient
    operator on a real vector -- so both the sum and the vector here are complex."""
    num_sites = 6
    pauli_sum = _mixed_pauli_sum(num_sites)
    rng = np.random.default_rng(0)
    dim = 2**num_sites
    vector = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    np.testing.assert_allclose(
        apply_pauli_sum(pauli_sum, vector), pauli_sum.to_matrix() @ vector, atol=1e-12
    )


@pytest.mark.parametrize("num_particles", [1, 3, 5])
def test_apply_pauli_sum_in_a_sector_matches_the_sector_matrix(num_particles):
    """The sector path must agree with to_sector_matrix, whose own reference is dense-slice."""
    num_sites = 6
    terms = []
    for site in range(num_sites - 1):
        for pauli in ("X", "Y"):
            string = ["I"] * num_sites
            string[site] = string[site + 1] = pauli
            terms.append(PauliString("".join(string), 0.5))
    pauli_sum = PauliSum(terms)

    indices = np.arange(2**num_sites, dtype=np.int64)
    sector = np.flatnonzero(np.bitwise_count(indices) == num_particles)

    rng = np.random.default_rng(1)
    vector = rng.normal(size=len(sector)) + 1j * rng.normal(size=len(sector))
    np.testing.assert_allclose(
        apply_pauli_sum(pauli_sum, vector, sector),
        pauli_sum.to_sector_matrix(sector) @ vector,
        atol=1e-12,
    )
