"""DynamicRound and DynamicCircuit: containers for multi-round DQC protocols."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from qaravan.core.base import NoiseModel, State
from qaravan.core.circuit import Circuit

Decoder = Callable[[str], Circuit]
"""Maps a measured bitstring (sorted meas_sites order) to the circuit applied next."""


@dataclass
class DynamicRound:
    """Measure meas_sites, then apply decoder(outcome); an empty list passes "" without measuring."""

    meas_sites: list[int] = field(default_factory=list)
    decoder: Decoder = field(
        default_factory=lambda: lambda _: Circuit([])
    )


@dataclass
class DynamicCircuit:
    """A sequence of DynamicRounds representing a full DQC protocol."""

    rounds: list[DynamicRound]

    def execute(
        self,
        initial_state: State,
        *,
        noise_model: NoiseModel | None = None,
        measurement_rng: np.random.Generator | int | None = None,
        noise_rng: np.random.Generator | int | None = None,
        observer: Callable[[int, State], None] | None = None,
        **simulator_kwargs,
    ) -> tuple[State, list[str]]:
        """Execute every round, returning the final state and measurement record."""
        measurement_generator = np.random.default_rng(measurement_rng)
        noise_generator = np.random.default_rng(noise_rng)

        state = initial_state
        outcomes: list[str] = []
        for round_index, dynamic_round in enumerate(self.rounds):
            if dynamic_round.meas_sites:
                state, outcome = state.measure_and_collapse(
                    dynamic_round.meas_sites, measurement_generator
                )
                outcomes.append(outcome)
            else:
                outcome = ""
            circuit = dynamic_round.decoder(outcome)
            if circuit.gates:
                state = state.apply(
                    circuit,
                    noise_model=noise_model,
                    noise_rng=noise_generator,
                    **simulator_kwargs,
                )
            if observer is not None:
                observer(round_index, state)
        return state, outcomes
