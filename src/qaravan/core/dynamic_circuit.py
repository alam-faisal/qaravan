"""DynamicRound and DynamicCircuit: containers for multi-round DQC protocols."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from qaravan.core.base import NoiseModel, State
from qaravan.core.circuit import Circuit


@dataclass
class DynamicRound:
    """One round of a DQC: measure meas_sites, then apply decoder(outcome).

    meas_sites=[] means no measurement; decoder receives "" and must return
    the circuit unconditionally (use lambda _: circuit for constant rounds).
    """

    meas_sites: list[int] = field(default_factory=list)
    decoder: Callable[[str], Circuit] = field(
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
        for round_index, rnd in enumerate(self.rounds):
            if rnd.meas_sites:
                state, outcome = state.measure_and_collapse(
                    rnd.meas_sites, measurement_generator
                )
                outcomes.append(outcome)
            else:
                outcome = ""
            circuit = rnd.decoder(outcome)
            if circuit.gates:
                if noise_model is None:
                    state = state.apply(circuit, **simulator_kwargs)
                else:
                    state = state.apply(
                        circuit,
                        noise_model=noise_model,
                        rng=noise_generator,
                        **simulator_kwargs,
                    )
            if observer is not None:
                observer(round_index, state)
        return state, outcomes
