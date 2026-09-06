"""DynamicRound and DynamicCircuit: containers for multi-round DQC protocols."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from qaravan.core.base import Simulator, State
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
        simulator: type[Simulator] | None = None,
        observer: Callable[[int, State], None] | None = None,
        measurement_rng: np.random.Generator | None = None,
        **simulator_kwargs,
    ) -> tuple[State, list[str]]:
        """Run every round on initial_state, returning (final_state, one outcome per measured
        round in round order).

        simulator evolves each round's circuit; it defaults to initial_state's own. Passing
        MonteCarloSimulator with a noise_model gives one noisy trajectory:

            dqc.execute(state, simulator=MonteCarloSimulator, noise_model=noise)

        measurement_rng is resolved once, so one seed fixes every round's outcome. A
        simulator's own randomness goes through simulator_kwargs as rng=, keeping the two
        streams independent -- a shared one would desynchronise the outcomes as soon as the
        simulator drew from it.

        observer(round_index, state) is called after each round, for per-round diagnostics the
        final state alone cannot give.
        """
        rng = np.random.default_rng(measurement_rng)
        if simulator is None:
            simulator = initial_state.default_simulator

        sv = initial_state
        outcomes: list[str] = []
        for round_index, rnd in enumerate(self.rounds):
            if rnd.meas_sites:
                sv, outcome = sv.measure_and_collapse(rnd.meas_sites, rng)
                outcomes.append(outcome)
            else:
                outcome = ""
            circuit = rnd.decoder(outcome)
            if circuit.gates:
                sv = simulator(circuit, sv, **simulator_kwargs).run()
            if observer is not None:
                observer(round_index, sv)
        return sv, outcomes
