"""Tests for Walkers(State) and MonteCarloSimulator."""

import numpy as np
import pytest
from qaravan.backends.monte_carlo import MonteCarloSimulator, Walkers
from qaravan.backends.statevector import Statevector, StatevectorSimulator
from qaravan.core.circuit import Circuit
from qaravan.core.gates import CNOT, RX, H
from qaravan.core.gates import Z as ZGate
from qaravan.core.noise import DepolarizingNoise
from qaravan.core.observables import PauliString

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bell_circuit() -> Circuit:
    return Circuit([H(0), CNOT([0, 1])], num_sites=2)


def bell_state() -> Statevector:
    return StatevectorSimulator(bell_circuit(), Statevector(bitstring="00")).run()


# ---------------------------------------------------------------------------
# Walkers
# ---------------------------------------------------------------------------


def test_walkers_measure_and_collapse_raises():
    walkers = Walkers([Statevector(bitstring="0")])
    with pytest.raises(NotImplementedError, match="MonteCarloSimulator"):
        walkers.measure_and_collapse([0])


def test_walkers_apply_raises():
    walkers = Walkers([Statevector(bitstring="0")])
    circ = Circuit([H(0)], num_sites=1)
    with pytest.raises(NotImplementedError):
        walkers.apply(circ)


def test_walkers_num_walkers():
    svs = [Statevector(bitstring="0") for _ in range(7)]
    walkers = Walkers(svs)
    assert walkers.num_walkers == 7
    assert walkers.num_sites == 1


def test_walkers_rdm_trace_one():
    circ = bell_circuit()
    init = Statevector(bitstring="00")
    mc = MonteCarloSimulator(
        circ, init, num_trajectories=10, rng=np.random.default_rng(0)
    )
    walkers = mc.run()
    rdm = walkers.rdm([0])
    assert abs(np.trace(rdm) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# MonteCarloSimulator — noiseless
# ---------------------------------------------------------------------------


def test_mc_noiseless_matches_sv():
    """Noiseless MC on Bell state: ⟨ZZ⟩ = 1 (Z⊗Z on (|00⟩+|11⟩)/√2)."""
    circ = bell_circuit()
    init = Statevector(bitstring="00")
    mc = MonteCarloSimulator(
        circ, init, num_trajectories=200, rng=np.random.default_rng(0)
    )
    walkers = mc.run()
    obs = PauliString("ZZ")
    assert abs(walkers.expectation(obs).real - 1.0) < 1e-10


def test_walkers_expectation_matches_sv_noiseless():
    """Single-trajectory noiseless MC agrees exactly with StatevectorSimulator."""
    circ = Circuit([RX(0, 0.7), RX(1, 1.3)], num_sites=2)
    init = Statevector(bitstring="00")
    sv = StatevectorSimulator(circ, init).run()
    mc = MonteCarloSimulator(
        circ, init, num_trajectories=1, rng=np.random.default_rng(0)
    )
    walkers = mc.run()
    obs = PauliString("ZI")
    assert abs(walkers.expectation(obs) - sv.expectation(obs)) < 1e-10


# ---------------------------------------------------------------------------
# MonteCarloSimulator — noisy
# ---------------------------------------------------------------------------


def test_mc_depolarizing_z_expectation():
    """Single qubit |0⟩ + global depol(p): ⟨Z⟩ = 1 - 4p/3 (analytical).

    Derivation: E(|0><0|) = diag(1-2p/3, 2p/3), so ⟨Z⟩ = 1 - 4p/3.
    Z|0⟩ = |0⟩, so the Z gate creates one circuit layer without changing the state.
    """
    p = 0.3
    expected = 1 - 4 * p / 3  # = 0.6
    circ = Circuit(
        [ZGate(0)], num_sites=1
    )  # one layer; Z|0⟩=|0⟩ leaves state unchanged
    init = Statevector(bitstring="0")
    noise = DepolarizingNoise(num_sites=1, error_rate=p)
    mc = MonteCarloSimulator(
        circ,
        init,
        noise_model=noise,
        num_trajectories=5000,
        rng=np.random.default_rng(42),
    )
    walkers = mc.run()
    z_exp = walkers.expectation(PauliString("Z")).real
    # variance per trajectory ≤ 1, std of mean ≤ 1/sqrt(5000) ≈ 0.014; use 4σ margin
    assert abs(z_exp - expected) < 0.06, f"⟨Z⟩ = {z_exp:.4f}, expected {expected:.4f}"


def test_mc_depolarizing_decays_zz_bell():
    """⟨ZZ⟩ on Bell state decays under per-layer global depolarizing.

    Noiseless: ⟨ZZ⟩ = 1. With noise it must be strictly less.
    """
    circ = bell_circuit()
    init = Statevector(bitstring="00")
    noise = DepolarizingNoise(num_sites=2, error_rate=0.5)
    mc = MonteCarloSimulator(
        circ,
        init,
        noise_model=noise,
        num_trajectories=2000,
        rng=np.random.default_rng(0),
    )
    walkers = mc.run()
    zz_noisy = walkers.expectation(PauliString("ZZ")).real
    assert zz_noisy < 1.0, f"⟨ZZ⟩ = {zz_noisy:.4f} should be < 1.0 under depolarizing"


def test_mc_reproducible_with_seed():
    """Same seed produces identical Walkers."""
    circ = bell_circuit()
    init = Statevector(bitstring="00")
    noise = DepolarizingNoise(num_sites=2, error_rate=0.2)
    kwargs = dict(noise_model=noise, num_trajectories=50)
    w1 = MonteCarloSimulator(circ, init, rng=np.random.default_rng(7), **kwargs).run()
    w2 = MonteCarloSimulator(circ, init, rng=np.random.default_rng(7), **kwargs).run()
    obs = PauliString("ZZ")
    assert abs(w1.expectation(obs) - w2.expectation(obs)) < 1e-14
