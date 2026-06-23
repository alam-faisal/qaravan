"""Tests for PauliNoise and DepolarizingNoise."""

import numpy as np
import pytest

from qaravan.core.noise import DepolarizingNoise, PauliNoise


# ---------------------------------------------------------------------------
# PauliNoise
# ---------------------------------------------------------------------------


def test_pauli_noise_probs_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        PauliNoise(["II", "XX"], [0.3, 0.4])


def test_pauli_noise_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        PauliNoise(["II", "XX"], [1.0])


def test_pauli_noise_sample_returns_string_from_distribution():
    """Empirical identity fraction should be close to 0.7."""
    rng = np.random.default_rng(0)
    noise = PauliNoise(["II", "XX", "YY", "ZZ"], [0.7, 0.1, 0.1, 0.1])
    samples = [noise.sample_error(rng) for _ in range(2000)]
    identity_fraction = samples.count("II") / 2000
    assert 0.63 < identity_fraction < 0.77


def test_pauli_noise_get_kraus_cptp():
    """CPTP condition: Σ_i K_i† K_i = I."""
    noise = PauliNoise(["I", "X", "Y", "Z"], [0.7, 0.1, 0.1, 0.1])
    kraus = noise.get_kraus()
    total = sum(k.conj().T @ k for k in kraus)
    assert np.allclose(total, np.eye(2), atol=1e-12)


def test_pauli_noise_get_kraus_multi_qubit_cptp():
    noise = PauliNoise(["II", "XX", "YY", "ZZ"], [0.7, 0.1, 0.1, 0.1])
    kraus = noise.get_kraus()
    total = sum(k.conj().T @ k for k in kraus)
    assert np.allclose(total, np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# DepolarizingNoise
# ---------------------------------------------------------------------------


def test_depolarizing_is_pauli_noise_subclass():
    assert issubclass(DepolarizingNoise, PauliNoise)


def test_depolarizing_identity_at_zero_rate():
    noise = DepolarizingNoise(num_sites=3, error_rate=0.0)
    rng = np.random.default_rng(42)
    for _ in range(200):
        assert noise.sample_error(rng) == "III"


def test_depolarizing_no_identity_at_full_rate():
    """error_rate=1 should never return the identity string."""
    noise = DepolarizingNoise(num_sites=2, error_rate=1.0)
    rng = np.random.default_rng(7)
    for _ in range(200):
        assert noise.sample_error(rng) != "II"


def test_depolarizing_sample_non_identity_rate():
    """Empirical non-identity rate must be within 3σ of error_rate."""
    p = 0.3
    noise = DepolarizingNoise(num_sites=1, error_rate=p)
    rng = np.random.default_rng(0)
    n_samples = 10_000
    non_identity = sum(1 for _ in range(n_samples) if noise.sample_error(rng) != "I")
    measured_rate = non_identity / n_samples
    sigma = np.sqrt(p * (1 - p) / n_samples)
    assert abs(measured_rate - p) < 4 * sigma, (
        f"measured rate = {measured_rate:.4f}, expected {p}, 4σ = {4*sigma:.4f}"
    )


def test_depolarizing_invalid_rate():
    with pytest.raises(ValueError):
        DepolarizingNoise(num_sites=2, error_rate=1.5)


def test_depolarizing_get_kraus_raises():
    noise = DepolarizingNoise(num_sites=2, error_rate=0.1)
    with pytest.raises(NotImplementedError):
        noise.get_kraus()


def test_depolarizing_sample_is_valid_pauli_string():
    """Every sample must be an n-qubit Pauli string over {I,X,Y,Z}."""
    noise = DepolarizingNoise(num_sites=4, error_rate=0.5)
    rng = np.random.default_rng(1)
    valid = set("IXYZ")
    for _ in range(500):
        s = noise.sample_error(rng)
        assert len(s) == 4
        assert all(c in valid for c in s)
