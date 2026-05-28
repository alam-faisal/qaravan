"""Circuit compilation via environment sweep variational state preparation."""

from __future__ import annotations

import numpy as np

from qaravan.applications.circuit_library import two_local_circuit
from qaravan.applications.run_context import RunContext
from qaravan.core.circuits import Circuit
from qaravan.core.gates import MatrixGate
from qaravan.core.base import State


def _environment_update(
    circ: Circuit,
    gate_idx: int,
    pre_state: State,
    post_state: State,
    init_state: State,
    target: State,
    direction: str,
) -> tuple[float, State, State]:
    """Update gate at gate_idx in-place; return (cost, new_pre_state, new_post_state).

    direction: "left" (decreasing k) or "right" (increasing k).
    cost = 1 - sum(singular values of environment) = 1 - |⟨target|U|init⟩| after update.
    """
    indices = circ.gates[gate_idx].indices
    env = pre_state.partial_overlap(post_state, skip=indices)

    x, s, yh = np.linalg.svd(env)
    new_mat = yh.conj().T @ x.conj().T  # polar factor: Y X†

    circ.gates[gate_idx] = MatrixGate(circ.gates[gate_idx].name, indices, new_mat)
    circ.layers = None  # invalidate layer cache

    if direction == "left":
        if gate_idx == 1:
            pre_state = init_state
        else:
            pre_state = pre_state.apply(Circuit([circ.gates[gate_idx - 1].dagger()]))
        post_state = post_state.apply(Circuit([circ.gates[gate_idx].dagger()]))
    else:  # "right"
        pre_state = pre_state.apply(Circuit([circ.gates[gate_idx]]))
        if gate_idx == len(circ) - 2:
            post_state = target
        else:
            post_state = post_state.apply(Circuit([circ.gates[gate_idx + 1]]))

    cost = 1.0 - float(np.sum(s))
    return cost, pre_state, post_state


def environment_state_prep(
    target: State,
    init_state: State,
    circuit: Circuit | None = None,
    skeleton: list[list[int]] | None = None,
    context: RunContext | None = None,
) -> tuple[Circuit, list[float]]:
    """Optimise a circuit to prepare target from init_state via the environment sweep.

    Exactly one of circuit or skeleton must be provided.
    Returns (optimised_circuit, cost_list) where:
      - cost_list[0] is infidelity before any optimization
      - subsequent entries are recorded after every gate update (fine-grained)
    Input circuit is never mutated.
    """
    if (circuit is None) == (skeleton is None):
        raise ValueError("Exactly one of circuit or skeleton must be provided.")

    context = context or RunContext()

    if circuit is not None:
        circ = circuit.copy()
    else:
        circ = two_local_circuit(skeleton)  # type: ignore[arg-type]

    # Initial full simulation
    ansatz = init_state.apply(circ)
    cost_list: list[float] = [1.0 - abs(target.overlap(ansatz))]

    # pre_state: full ansatz with last gate undone
    pre_state = ansatz.apply(Circuit([circ.gates[-1].dagger()]))
    post_state = target

    sweep_costs: list[float] = []

    for sweep in range(context.max_iter):
        # Left sweep: gate_idx from len(circ)-1 down to 1
        for gate_idx in reversed(range(1, len(circ))):
            cost, pre_state, post_state = _environment_update(
                circ, gate_idx, pre_state, post_state, init_state, target, "left"
            )
            cost_list.append(cost)

        # Right sweep: gate_idx from 0 up to len(circ)-2
        for gate_idx in range(len(circ) - 1):
            cost, pre_state, post_state = _environment_update(
                circ, gate_idx, pre_state, post_state, init_state, target, "right"
            )
            cost_list.append(cost)

        sweep_costs.append(cost_list[-1])
        if context.should_stop(sweep_costs, sweep + 1):
            break

    return circ, cost_list
