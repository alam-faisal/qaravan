"""Circuit compilation via environment sweep variational state preparation."""

from __future__ import annotations

import numpy as np

from typing import Callable

from qaravan.applications.circuit_library import (
    bell_basis_circuit,
    ghz_cluster_prep_circuit,
    two_local_circuit,
)
from qaravan.applications.run_context import RunContext
from qaravan.backends.statevector import Statevector
from qaravan.core.circuits import Circuit
from qaravan.core.gates import MatrixGate, X, Z
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


# ---------------------------------------------------------------------------
# GHZ state preparation via cluster fusion
# ---------------------------------------------------------------------------


def _kept_sites(cluster_idx: int, cluster_size: int, num_clusters: int) -> list[int]:
    """Non-fusion site indices for cluster cluster_idx."""
    if cluster_idx == 0:
        return list(range(0, cluster_size - 1))
    if cluster_idx == num_clusters - 1:
        return list(
            range((num_clusters - 1) * cluster_size + 1, num_clusters * cluster_size)
        )
    return list(
        range(cluster_idx * cluster_size + 1, (cluster_idx + 1) * cluster_size - 1)
    )


def build_ghz_decoder(
    cluster_size: int,
    num_clusters: int,
    total_qubits: int,
    use_cancellations: bool = False,
) -> Callable[[str], Circuit]:
    """builds decoder for the GHZ fusion measurement.

    Returns decoder(outcome) → Circuit.
    outcome: bitstring of length 2*(num_clusters-1); bits 2*i, 2*i+1 are
    phase_bit and flip_bit for boundary i.
    use_cancellations: fold repeated Z(0) and X(site) corrections before returning.
    """
    num_boundaries = num_clusters - 1

    def decoder(outcome: str) -> Circuit:
        phase_bits = [int(outcome[2 * i]) for i in range(num_boundaries)]
        flip_bits = [int(outcome[2 * i + 1]) for i in range(num_boundaries)]

        gates = []
        if use_cancellations:
            if sum(phase_bits) % 2 == 1:
                gates.append(Z(0))
            for cluster_idx in range(1, num_clusters):
                if sum(flip_bits[:cluster_idx]) % 2 == 1:
                    for site in _kept_sites(cluster_idx, cluster_size, num_clusters):
                        gates.append(X(site))
        else:
            for boundary_idx in range(num_boundaries):
                if flip_bits[boundary_idx]:
                    for cluster_idx in range(boundary_idx + 1, num_clusters):
                        for site in _kept_sites(
                            cluster_idx, cluster_size, num_clusters
                        ):
                            gates.append(X(site))
                if phase_bits[boundary_idx]:
                    gates.append(Z(0))

        return Circuit(gates, num_sites=total_qubits)

    return decoder


def ghz_via_fusion(
    n: int, cluster_size: int = 3, use_cancellations: bool = False
) -> tuple[Statevector, str]:
    """Prepare n-qubit GHZ via mid-circuit-measurement fusion of cluster_size clusters.
    Requires cluster_size > 2 and (n - 2) % (cluster_size - 2) == 0.
    Z corrections always target qubit 0 (phase anchor).
    """
    if cluster_size <= 2:
        raise ValueError(f"cluster_size must be > 2; got {cluster_size}")
    if (n - 2) % (cluster_size - 2) != 0:
        raise ValueError(
            f"(n-2)={(n - 2)} must be divisible by (cluster_size-2)={(cluster_size - 2)}"
        )

    num_clusters = (n - 2) // (cluster_size - 2)
    total_qubits = num_clusters * cluster_size

    # prepare GHZ on clusters
    prep_circuit = Circuit([], num_sites=total_qubits)
    for cluster_idx in range(num_clusters):
        cluster_sites = list(
            range(cluster_idx * cluster_size, (cluster_idx + 1) * cluster_size)
        )
        prep_circuit = prep_circuit + ghz_cluster_prep_circuit(
            cluster_sites, total_qubits
        )

    sv = Statevector(bitstring="0" * total_qubits).apply(prep_circuit)

    # Bell-measurement of boundary qubit pairs
    boundary_qubits: list[int] = []
    bell_circuit = Circuit([], num_sites=total_qubits)
    for boundary_idx in range(num_clusters - 1):
        left_qubit = (boundary_idx + 1) * cluster_size - 1
        right_qubit = (boundary_idx + 1) * cluster_size
        boundary_qubits.extend([left_qubit, right_qubit])
        bell_circuit = bell_circuit + bell_basis_circuit(
            left_qubit, right_qubit, total_qubits
        )

    sv = sv.apply(bell_circuit)
    sv, outcome = sv.measure_and_collapse(boundary_qubits)

    # decode outcome → correction circuit
    decoder = build_ghz_decoder(
        cluster_size, num_clusters, total_qubits, use_cancellations
    )
    correction = decoder(outcome)
    if correction.gates:
        sv = sv.apply(correction)

    return sv, outcome
