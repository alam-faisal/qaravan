"""Concrete NoiseModel subclasses: PauliNoise, DepolarizingNoise."""

from __future__ import annotations
from typing import Literal
import numpy as np
from qaravan.core.base import NoiseModel
from qaravan.core.observables import PauliString

NoisePlacement = Literal["layer", "circuit"]


class PauliNoise(NoiseModel):
    """Explicit Pauli channel sampled after each layer or once after the circuit."""

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
        self.placement = placement

    def sample_error(self, rng: np.random.Generator | None = None) -> str:
        """Draw one Pauli string from the channel distribution."""
        rng = np.random.default_rng(rng)
        idx = rng.choice(len(self.strings), p=self.probs)
        return self.strings[idx]

    def get_kraus(self, *args, **kwargs) -> list[np.ndarray]:
        """Return one √p_i P_i Kraus operator per channel term."""
        return [
            np.sqrt(p) * PauliString(s).to_matrix()
            for s, p in zip(self.strings, self.probs)
        ]


class DepolarizingNoise(PauliNoise):
    """Global depolarizing channel sampled in O(n) without enumerating 4^n strings."""

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
        self.placement = placement

    def sample_error(self, rng: np.random.Generator | None = None) -> str:
        """Draw identity with probability 1-p or a uniform nonidentity Pauli with probability p."""
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
