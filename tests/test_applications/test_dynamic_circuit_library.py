"""Tests for dynamic_circuit_library: _kept_sites, build_ghz_decoder, ghz_fusion_dqc."""

import numpy as np
import pytest
from scipy.linalg import expm
from qaravan.applications.dynamic_circuit_execution import (
    execute_dynamic_circuit_ensemble,
)
from qaravan.applications.dynamic_circuit_library import (
    _kept_sites,
    build_ghz_decoder,
    collisional_dqc,
    dissipative_ising_layer,
    ghz_fusion_dqc,
    interleaved_collisional_layout,
    qcp_layer,
    reset_decoder,
)
from qaravan.backends.statevector import Statevector
from qaravan.core.circuit import Circuit
from qaravan.core.dynamic_circuit import DynamicCircuit, DynamicRound
from qaravan.core.gates import H, X
from qaravan.core.noise import DepolarizingNoise
from qaravan.core.observable import NumberOperator

INV_SQRT2 = 1.0 / np.sqrt(2)


def _ghz_fidelity(sv: Statevector, fusion_sites: list[int], outcome: str) -> float:
    """F = |⟨GHZ_n|ψ_kept⟩|² using drop_sites to extract the pure kept-qubit state."""
    pure_kept = sv.drop_sites(fusion_sites, outcome)
    n = pure_kept.num_sites
    ghz = np.zeros(2**n)
    ghz[0] = ghz[-1] = INV_SQRT2
    return float(abs(np.dot(ghz, pure_kept.to_array())) ** 2)


# ---------------------------------------------------------------------------
# _kept_sites
# ---------------------------------------------------------------------------


def test_kept_sites_first_cluster():
    """First cluster (cluster_idx=0): sites 0..cluster_size-2."""
    assert _kept_sites(0, 3, 4) == [0, 1]
    assert _kept_sites(0, 4, 3) == [0, 1, 2]


def test_kept_sites_last_cluster():
    """Last cluster: sites (C-1)*k+1 .. C*k-1."""
    # 4 clusters of size 3: last cluster is sites 9,10,11 → kept = [10,11]
    assert _kept_sites(3, 3, 4) == [10, 11]


def test_kept_sites_interior_cluster():
    """Interior cluster: single site cluster_idx*k+1 for cluster_size=3."""
    # cluster_idx=1, cluster_size=3, num_clusters=4 → site 4
    assert _kept_sites(1, 3, 4) == [4]
    # cluster_idx=2, cluster_size=3, num_clusters=4 → site 7
    assert _kept_sites(2, 3, 4) == [7]


# ---------------------------------------------------------------------------
# build_ghz_decoder — output validation
# ---------------------------------------------------------------------------


def test_build_ghz_decoder_empty_outcome_gives_no_correction():
    """All-zero outcome '00' → no gates (no corrections needed)."""
    decoder = build_ghz_decoder(cluster_size=3, num_clusters=2, total_qubits=6)
    circ = decoder("00")
    assert circ.gates == []


def test_build_ghz_decoder_cancellations_agree_with_naive():
    """use_cancellations=True and False give same final state for n=4 (all 4 outcomes).

    Uses statevector application rather than full unitary matrix — avoids 4096×4096 matrices.
    Catches: cancellation folding giving wrong sign or wrong qubit.
    Misses: both paths wrong the same way.
    """
    cs, num_clusters, total_qubits = 3, 2, 6
    d_naive = build_ghz_decoder(cs, num_clusters, total_qubits, use_cancellations=False)
    d_cancel = build_ghz_decoder(cs, num_clusters, total_qubits, use_cancellations=True)
    sv_ref = Statevector(num_sites=total_qubits, random_seed=7)
    for outcome in ["00", "01", "10", "11"]:
        sv_naive = sv_ref.apply(d_naive(outcome)) if d_naive(outcome).gates else sv_ref
        sv_cancel = (
            sv_ref.apply(d_cancel(outcome)) if d_cancel(outcome).gates else sv_ref
        )
        np.testing.assert_allclose(
            sv_naive.to_array(),
            sv_cancel.to_array(),
            atol=1e-10,
            err_msg=f"outcome={outcome}: naive and cancel disagree",
        )


# ---------------------------------------------------------------------------
# ghz_fusion_dqc — input validation
# ---------------------------------------------------------------------------


def test_ghz_fusion_dqc_raises_for_small_cluster_size():
    """cluster_size <= 2 raises ValueError."""
    with pytest.raises(ValueError):
        ghz_fusion_dqc(4, cluster_size=2)


def test_ghz_fusion_dqc_raises_for_non_integer_num_clusters():
    """(n-2) not divisible by (cluster_size-2) raises ValueError."""
    with pytest.raises(ValueError):
        ghz_fusion_dqc(5, cluster_size=4)


# ---------------------------------------------------------------------------
# ghz_fusion_dqc — structure
# ---------------------------------------------------------------------------


def test_ghz_fusion_dqc_has_two_rounds():
    """ghz_fusion_dqc always returns a DynamicCircuit with exactly 2 rounds."""
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    assert isinstance(dqc, DynamicCircuit)
    assert len(dqc.rounds) == 2


def test_ghz_fusion_dqc_round0_no_measurement():
    """Round 0 has meas_sites=[]."""
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    assert dqc.rounds[0].meas_sites == []


def test_ghz_fusion_dqc_round1_meas_sites():
    """Round 1 meas_sites == boundary qubit indices.

    n=4, cluster_size=3: 2 clusters, 1 boundary at qubits [2,3].
    n=6, cluster_size=3: 4 clusters, 3 boundaries at [2,3,5,6,8,9].
    Catches: off-by-one or wrong boundary qubit formula.
    """
    dqc4 = ghz_fusion_dqc(4, cluster_size=3)
    assert dqc4.rounds[1].meas_sites == [2, 3]

    dqc6 = ghz_fusion_dqc(6, cluster_size=3)
    assert dqc6.rounds[1].meas_sites == [2, 3, 5, 6, 8, 9]


# ---------------------------------------------------------------------------
# ghz_fusion_dqc — physics correctness
# ---------------------------------------------------------------------------


def test_ghz_fusion_dqc_4qubit_fidelity():
    """n=4, cluster_size=3: 20 runs, fidelity with |GHZ_4⟩ on kept sites = 1.0.

    Fusion sites: {2,3}. Kept: {0,1,4,5}.
    Catches: wrong correction for any of the 4 outcomes.
    """
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    init = Statevector(bitstring="000000")
    for _ in range(20):
        sv, outcomes = dqc.execute(init)
        assert np.isclose(_ghz_fidelity(sv, [2, 3], outcomes[0]), 1.0, atol=1e-10)


def test_ghz_fusion_dqc_4qubit_all_outcomes_covered():
    """100 runs of n=4 must yield all 4 outcome strings.

    Catches: correction logic that silently fails for specific outcomes.
    """
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    init = Statevector(bitstring="000000")
    seen: set[str] = set()
    for _ in range(100):
        _, outcomes = dqc.execute(init)
        seen.add(outcomes[0])
    assert seen == {"00", "01", "10", "11"}


def test_ghz_fusion_dqc_6qubit_fidelity():
    """n=6, cluster_size=3: 20 runs, fidelity = 1.0, outcome length = 6.

    Fusion sites: {2,3,5,6,8,9}. 3 boundaries × 2 bits = 6-bit outcome.
    Catches: decoder error for multi-boundary case.
    """
    dqc = ghz_fusion_dqc(6, cluster_size=3)
    init = Statevector(bitstring="0" * 12)
    for _ in range(20):
        sv, outcomes = dqc.execute(init)
        assert len(outcomes[0]) == 6
        assert np.isclose(
            _ghz_fidelity(sv, [2, 3, 5, 6, 8, 9], outcomes[0]), 1.0, atol=1e-10
        )


def test_ghz_fusion_dqc_use_cancellations_same_physics():
    """use_cancellations=True gives fidelity 1.0 — same final state as naive.

    Catches: cancellation folding introducing a sign or qubit error.
    """
    init = Statevector(bitstring="000000")
    for _ in range(20):
        dqc = ghz_fusion_dqc(4, cluster_size=3, use_cancellations=True)
        sv, outcomes = dqc.execute(init)
        assert np.isclose(_ghz_fidelity(sv, [2, 3], outcomes[0]), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Collisional models
# ---------------------------------------------------------------------------

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.diag([1.0, -1.0]).astype(complex)
_I = np.eye(2, dtype=complex)
_N = (_I - _Z) / 2


def _embed(factors: dict[int, np.ndarray], num_sites: int) -> np.ndarray:
    """Kron single-site factors in site order, identity where unspecified."""
    matrix = factors.get(0, _I)
    for site in range(1, num_sites):
        matrix = np.kron(matrix, factors.get(site, _I))
    return matrix


def test_interleaved_collisional_layout():
    system_sites, ancilla_sites, coupling_pairs = interleaved_collisional_layout(3)
    assert system_sites == [0, 2, 4]
    assert ancilla_sites == [1, 3, 5]
    assert coupling_pairs == [(0, 1), (2, 3), (4, 5)]


@pytest.mark.parametrize("dt", [0.05, 0.02])
def test_dissipative_ising_layer_is_first_order_trotter(dt):
    system_sites, _, _ = interleaved_collisional_layout(3)
    num_sites, omega, field_v = 6, 0.8, 0.6

    exact_generator = np.zeros((2**num_sites, 2**num_sites), dtype=complex)
    for site in system_sites:
        exact_generator += omega * _embed({site: _X}, num_sites)
    for left, right in zip(system_sites, system_sites[1:]):
        exact_generator += field_v * _embed({left: _Z, right: _Z}, num_sites)

    layer = dissipative_ising_layer(system_sites, omega, field_v, dt)
    approx = Circuit(layer, num_sites=num_sites).to_matrix()
    error = np.linalg.norm(approx - expm(-1j * dt * exact_generator), ord="fro")
    assert error < 20 * dt**2


@pytest.mark.parametrize("dt", [0.05, 0.02])
def test_qcp_layer_is_first_order_trotter(dt):
    system_sites, _, _ = interleaved_collisional_layout(3)
    num_sites, omega = 6, 0.8

    exact_generator = np.zeros((2**num_sites, 2**num_sites), dtype=complex)
    for left, right in zip(system_sites, system_sites[1:]):
        exact_generator += (omega / 2) * _embed({left: _X, right: _N}, num_sites)
        exact_generator += (omega / 2) * _embed({left: _N, right: _X}, num_sites)

    layer = qcp_layer(system_sites, omega, dt)
    approx = Circuit(layer, num_sites=num_sites).to_matrix()
    error = np.linalg.norm(approx - expm(-1j * dt * exact_generator), ord="fro")
    assert error < 20 * dt**2


def test_collisional_dqc_round_structure():
    _, ancilla_sites, coupling_pairs = interleaved_collisional_layout(2)
    dqc = collisional_dqc(
        [], ancilla_sites, coupling_pairs, 0.3, num_cycles=4, num_sites=4
    )
    assert len(dqc.rounds) == 5
    assert dqc.rounds[0].meas_sites == []
    assert all(rnd.meas_sites == ancilla_sites for rnd in dqc.rounds[1:])


def test_collisional_dqc_resets_ancillas_to_zero():
    """Ancillas start in |1> with no coupling, so round 1 must measure '11' and flip them
    back. A reset that wrote |1>, or skipped the last cycle, would leave them occupied.
    """
    num_system = 2
    _, ancilla_sites, coupling_pairs = interleaved_collisional_layout(num_system)
    dqc = collisional_dqc(
        [], ancilla_sites, coupling_pairs, 0.0, num_cycles=2, num_sites=4
    )

    final, outcomes = dqc.execute(Statevector(bitstring="11" * num_system))
    assert outcomes[0] == "1" * num_system
    assert outcomes[1] == "0" * num_system
    assert final.expectation(NumberOperator(4, qubits=ancilla_sites)).real < 1e-9


def test_collisional_dqc_with_zero_coupling_never_decays():
    num_system = 2
    system_sites, ancilla_sites, coupling_pairs = interleaved_collisional_layout(
        num_system
    )
    dqc = collisional_dqc(
        [], ancilla_sites, coupling_pairs, 0.0, num_cycles=5, num_sites=4
    )
    final, outcomes = dqc.execute(Statevector(bitstring="10" * num_system))
    assert all(outcome == "0" * num_system for outcome in outcomes)
    system_number = NumberOperator(4, qubits=system_sites)
    assert np.isclose(final.expectation(system_number).real, num_system)


def test_collisional_activity_matches_the_closed_form():
    """omega = V = 0: pairs decouple, so E[k_m] = p (1-p)^(m-1) with p = sin^2(phi).

    Catches a reset that writes |1> (activity would not decay), wrong ancilla indices,
    and a coupling angle off by a factor of two. Seeded, so the 4-sigma band is a fixed
    bound rather than a per-run coin flip.
    """
    num_system, phi, num_cycles, num_shots = 2, 0.5, 10, 800
    _, ancilla_sites, coupling_pairs = interleaved_collisional_layout(num_system)
    dqc = collisional_dqc(
        [], ancilla_sites, coupling_pairs, phi, num_cycles, 2 * num_system
    )
    initial = Statevector(bitstring="10" * num_system)

    rng = np.random.default_rng(0)
    activities = np.zeros((num_shots, num_cycles))
    for shot in range(num_shots):
        _, outcomes = dqc.execute(initial, measurement_rng=rng)
        for cycle, outcome in enumerate(outcomes):
            activities[shot, cycle] = outcome.count("1") / num_system

    probability = np.sin(phi) ** 2
    expected = probability * (1 - probability) ** np.arange(num_cycles)
    sigma = np.sqrt(expected * (1 - expected) / (num_shots * num_system))
    assert np.all(np.abs(activities.mean(axis=0) - expected) < 4 * sigma + 1e-9)


def test_reset_decoder_drops_straight_into_a_dynamic_round():
    """The point of the factory form: usable as decoder= with no wrapping."""
    qubits = [0, 2]
    dqc = DynamicCircuit(
        [DynamicRound(meas_sites=qubits, decoder=reset_decoder(qubits, num_sites=3))]
    )
    final, outcomes = dqc.execute(Statevector(bitstring="111"))
    assert outcomes == ["11"]
    assert np.isclose(final.expectation(NumberOperator(3, qubits=qubits)).real, 0.0)
    # the unmeasured qubit is untouched
    assert np.isclose(final.expectation(NumberOperator(3, qubits=[1])).real, 1.0)


def test_reset_decoder_leaves_zeros_alone():
    decoder = reset_decoder([0, 1], num_sites=2)
    assert decoder("00").gates == []
    assert len(decoder("10").gates) == 1
    assert decoder("10").gates[0].indices == [0]


# ---------------------------------------------------------------------------
# noisy_dqc_trajectories
# ---------------------------------------------------------------------------


def _seeded_regression_dqc() -> DynamicCircuit:
    reset_to_plus = Circuit([X(0), H(0)], num_sites=1)
    rounds = [DynamicRound(decoder=lambda _: Circuit([H(0)], num_sites=1))]
    rounds.extend(
        DynamicRound(meas_sites=[0], decoder=lambda _: reset_to_plus) for _ in range(5)
    )
    return DynamicCircuit(rounds)


def test_seeded_noisy_scalar_path_regression():
    """Freeze noise/measurement draw ordering across the Monte Carlo refactor."""
    final, outcomes = _seeded_regression_dqc().execute(
        Statevector(bitstring="0"),
        measurement_rng=np.random.default_rng(4),
        noise_model=DepolarizingNoise(1, 0.35, placement="circuit"),
        noise_rng=np.random.default_rng(9),
    )

    assert outcomes == ["1", "1", "1", "0", "1"]
    np.testing.assert_allclose(
        final.to_array(),
        np.array([-1j, -1j]) / np.sqrt(2),
        atol=1e-12,
    )


def test_seeded_noisy_ensemble_regression():
    """Freeze complete records and final paths across the DQC API migration."""
    walkers, records = execute_dynamic_circuit_ensemble(
        _seeded_regression_dqc(),
        Statevector(bitstring="0"),
        4,
        noise_model=DepolarizingNoise(1, 0.35, placement="circuit"),
        rng=np.random.default_rng(3),
    )

    assert records == [
        ["1", "0", "1", "1", "0"],
        ["0", "0", "1", "1", "1"],
        ["1", "0", "1", "0", "0"],
        ["1", "0", "0", "0", "0"],
    ]
    expected = np.array(
        [
            [1, 1],
            [-1, -1],
            [-1, 1],
            [-1j, 1j],
        ],
        dtype=complex,
    ) / np.sqrt(2)
    actual = np.array([walker.to_array() for walker in walkers._walkers])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_zero_noise_reproduces_the_noiseless_run_exactly():
    """error_rate=0 must be a no-op, outcomes and amplitudes alike.

    Measurement and noise use separately spawned streams, so a zero-rate sample_error -- which
    still draws -- cannot shift the measurement record. Catches any leakage between the two.
    """
    num_system = 2
    _, ancilla_sites, coupling_pairs = interleaved_collisional_layout(num_system)
    num_sites = 2 * num_system
    dqc = collisional_dqc([], ancilla_sites, coupling_pairs, 0.5, 6, num_sites)
    initial = Statevector(bitstring="10" * num_system)

    measurement_rng, _ = np.random.default_rng(0).spawn(2)
    clean_state, clean_outcomes = dqc.execute(initial, measurement_rng=measurement_rng)

    walkers, records = execute_dynamic_circuit_ensemble(
        dqc,
        initial,
        1,
        noise_model=DepolarizingNoise(num_sites, 0.0),
        rng=np.random.default_rng(0),
    )
    assert records[0] == clean_outcomes
    assert np.allclose(walkers.rdm([0]), clean_state.rdm([0]))


def test_ghz_fidelity_degrades_with_error_rate():
    """Catches noise sampled once and reused, or applied to the wrong register.

    Fidelity comes from the kept-qubit density matrix, not drop_sites: noise lands after the
    measurement too, so the fusion qubits no longer hold the outcome that was recorded.
    """
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    initial = Statevector(bitstring="000000")
    kept_sites = [0, 1, 4, 5]
    ghz = np.zeros(2 ** len(kept_sites))
    ghz[0] = ghz[-1] = INV_SQRT2

    fidelities = []
    for error_rate in [0.0, 0.05, 0.2]:
        walkers, _ = execute_dynamic_circuit_ensemble(
            dqc,
            initial,
            40,
            noise_model=DepolarizingNoise(6, error_rate),
            rng=np.random.default_rng(5),
        )
        density = walkers.rdm(kept_sites)
        fidelities.append(float(np.real(ghz @ density @ ghz)))

    assert np.isclose(fidelities[0], 1.0, atol=1e-10)
    assert fidelities[0] > fidelities[1] > fidelities[2]


def test_noisy_trajectories_explore_different_branches():
    """The ensemble is over noise and outcomes jointly, not noise alone."""
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    _, records = execute_dynamic_circuit_ensemble(
        dqc,
        Statevector(bitstring="000000"),
        40,
        noise_model=DepolarizingNoise(6, 0.05),
        rng=np.random.default_rng(2),
    )
    assert len({tuple(record) for record in records}) > 1


def test_noisy_dqc_returns_one_record_per_trajectory():
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    walkers, records = execute_dynamic_circuit_ensemble(
        dqc,
        Statevector(bitstring="000000"),
        7,
        noise_model=DepolarizingNoise(6, 0.1),
        rng=np.random.default_rng(0),
    )
    assert walkers.num_walkers == 7
    assert len(records) == 7


def test_noisy_dqc_is_reproducible():
    dqc = ghz_fusion_dqc(4, cluster_size=3)
    initial = Statevector(bitstring="000000")
    kwargs = dict(
        num_trajectories=5,
        noise_model=DepolarizingNoise(6, 0.1),
    )
    first = execute_dynamic_circuit_ensemble(
        dqc, initial, rng=np.random.default_rng(3), **kwargs
    )[1]
    second = execute_dynamic_circuit_ensemble(
        dqc, initial, rng=np.random.default_rng(3), **kwargs
    )[1]
    assert first == second
