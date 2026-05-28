"""DynamicRound and DynamicCircuit: containers for multi-round DQC protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from qaravan.core.base import State
from qaravan.core.circuits import Circuit


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

    def run(self, init_state: State) -> tuple[State, list[str]]:
        """Execute all rounds on init_state.

        Returns (final_state, outcomes) where outcomes has one entry per round
        with non-empty meas_sites, in round order.
        """
        sv = init_state
        outcomes: list[str] = []
        for rnd in self.rounds:
            if rnd.meas_sites:
                sv, outcome = sv.measure_and_collapse(rnd.meas_sites)
                outcomes.append(outcome)
            else:
                outcome = ""
            circuit = rnd.decoder(outcome)
            if circuit.gates:
                sv = sv.apply(circuit)
        return sv, outcomes
