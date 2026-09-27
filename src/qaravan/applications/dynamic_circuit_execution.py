"""Execution helpers for dynamic-circuit trajectory ensembles."""

from __future__ import annotations
from typing import Callable
import numpy as np
from qaravan.backends.monte_carlo import Walkers
from qaravan.core.base import NoiseModel, State
from qaravan.core.dynamic_circuit import DynamicCircuit


def execute_dynamic_circuit_ensemble(
    dynamic_circuit: DynamicCircuit,
    initial_state: State,
    num_trajectories: int,
    rng: np.random.Generator | int | None = None,
    observer: Callable[[int, State], None] | None = None,
    *,
    noise_model: NoiseModel | None = None,
    **simulator_kwargs,
) -> tuple[Walkers, list[list[str]]]:
    """Execute independent dynamic-circuit trajectories and retain every record."""
    measurement_rng, noise_rng = np.random.default_rng(rng).spawn(2)
    final_states = []
    records = []
    for _ in range(num_trajectories):
        final_state, outcomes = dynamic_circuit.execute(
            initial_state,
            noise_model=noise_model,
            measurement_rng=measurement_rng,
            noise_rng=noise_rng,
            observer=observer,
            **simulator_kwargs,
        )
        final_states.append(final_state)
        records.append(outcomes)
    return Walkers(final_states), records
