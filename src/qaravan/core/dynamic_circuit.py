"""DynamicRound and DynamicCircuit: containers for multi-round DQC protocols."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from qaravan.core.base import State
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

    def run(
        self,
        init_state: State,
        observer: Callable[[int, State], None] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[State, list[str]]:
        """Execute all rounds on init_state.

        Returns (final_state, outcomes) where outcomes has one entry per round
        with non-empty meas_sites, in round order.

        observer(round_index, state) is called after each round, for per-round diagnostics
        that the final state alone cannot give.

        rng seeds every round's measurement; it is resolved once here, so one seed makes the
        whole protocol reproducible.
        """
        rng = np.random.default_rng(rng)
        sv = init_state
        outcomes: list[str] = []
        for round_index, rnd in enumerate(self.rounds):
            if rnd.meas_sites:
                sv, outcome = sv.measure_and_collapse(rnd.meas_sites, rng)
                outcomes.append(outcome)
            else:
                outcome = ""
            circuit = rnd.decoder(outcome)
            if circuit.gates:
                sv = sv.apply(circuit)
            if observer is not None:
                observer(round_index, sv)
        return sv, outcomes
