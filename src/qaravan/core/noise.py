"""Concrete NoiseModel subclasses: PauliNoise, DepolarizingNoise."""

from __future__ import annotations

from typing import Literal
import numpy as np
from qaravan.core.base import NoiseModel
from qaravan.core.observables import PauliString


NoisePlacement = Literal["layer", "circuit"]


def _validate_placement(placement: NoisePlacement) -> NoisePlacement:
    if placement not in ("layer", "circuit"):
        raise ValueError(
            f"placement must be 'layer' or 'circuit'; got {placement!r}"
        )
    return placement


class PauliNoise(NoiseModel):
    """Pauli channel with explicit strings and probabilities.

    strings: n-qubit Pauli strings e.g. ["II", "XX", "ZZ"]
    probs: matching probabilities, must sum to 1 within 1e-6
    """

    gate_dependent: bool = False

    def __init__(
        self,
        strings: list[str],
        probs: list[float],
        *,
        placement: NoisePlacement = "layer",
    ):
        if len(strings) != len(probs):
            raise ValueError(
                f"strings and probs must have the same length; "
                f"got {len(strings)} and {len(probs)}"
            )
        if abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError(f"Probabilities must sum to 1; got {sum(probs):.8f}")
        self.strings = [s.upper() for s in strings]
        self.probs = list(probs)
        self.placement = _validate_placement(placement)

    def sample_error(self, rng: np.random.Generator | None = None) -> str:
        """Multinomial draw; returns one Pauli string from the channel distribution."""
        rng = np.random.default_rng(rng)
        idx = rng.choice(len(self.strings), p=self.probs)
        return self.strings[idx]

    def get_kraus(self, *args, **kwargs) -> list[np.ndarray]:
        """Kraus operators: sqrt(p_i) * Pauli_i matrix for each term."""
        return [
            np.sqrt(p) * PauliString(s).to_matrix()
            for s, p in zip(self.strings, self.probs)
        ]


class DepolarizingNoise(PauliNoise):
    """Global n-qubit depolarizing channel.

    E(ρ) = (1-p)ρ + p/(4^n - 1) Σ_{P ≠ I^⊗n} P ρ P†

    Subclasses PauliNoise (IS-A Pauli channel) but overrides __init__ and
    sample_error to avoid enumerating 4^n strings. O(n) per sample.
    """

    def __init__(
        self,
        num_sites: int,
        error_rate: float,
        *,
        placement: NoisePlacement = "layer",
    ):
        if not 0.0 <= error_rate <= 1.0:
            raise ValueError(f"error_rate must be in [0, 1]; got {error_rate}")
        self.num_sites = num_sites
        self.error_rate = error_rate
        self.placement = _validate_placement(placement)

    def sample_error(self, rng: np.random.Generator | None = None) -> str:
        """Returns identity string with prob (1-p), uniform non-identity Pauli with prob p."""
        rng = np.random.default_rng(rng)
        if rng.random() >= self.error_rate:
            return "I" * self.num_sites
        identity = "I" * self.num_sites
        paulis = ["I", "X", "Y", "Z"]
        while True:
            string = "".join(rng.choice(paulis, size=self.num_sites))
            if string != identity:
                return string

    def get_kraus(self, *args, **kwargs) -> list[np.ndarray]:
        raise NotImplementedError(
            "DepolarizingNoise.get_kraus is deferred until DensityMatrixSimulator (v0.2.3)."
        )
