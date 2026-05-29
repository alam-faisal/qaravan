"""Builders for DynamicCircuit instances for concrete DQC protocols."""

from __future__ import annotations

from typing import Callable

from qaravan.applications.circuit_library import (
    bell_basis_circuit,
    ghz_cluster_prep_circuit,
)
from qaravan.core.circuit import Circuit
from qaravan.core.dynamic_circuit import DynamicCircuit, DynamicRound
from qaravan.core.gates import X, Z


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
    """Decoder for the GHZ fusion measurement.

    Returns decoder(outcome) -> Circuit.
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


def ghz_fusion_dqc(
    n: int,
    cluster_size: int = 3,
    use_cancellations: bool = False,
) -> DynamicCircuit:
    """DynamicCircuit for n-qubit GHZ via cluster fusion.

    Round 0: constant decoder applies prep_circuit + bell_circuit (no measurement).
    Round 1: measures boundary qubits, decoder returns Pauli correction.
    Requires cluster_size > 2 and (n - 2) % (cluster_size - 2) == 0.
    """
    if cluster_size <= 2:
        raise ValueError(f"cluster_size must be > 2; got {cluster_size}")
    if (n - 2) % (cluster_size - 2) != 0:
        raise ValueError(
            f"(n-2)={(n - 2)} must be divisible by (cluster_size-2)={(cluster_size - 2)}"
        )

    num_clusters = (n - 2) // (cluster_size - 2)
    total_qubits = num_clusters * cluster_size

    prep_circuit = Circuit([], num_sites=total_qubits)
    for cluster_idx in range(num_clusters):
        cluster_sites = list(
            range(cluster_idx * cluster_size, (cluster_idx + 1) * cluster_size)
        )
        prep_circuit = prep_circuit + ghz_cluster_prep_circuit(
            cluster_sites, total_qubits
        )

    boundary_qubits: list[int] = []
    bell_circuit = Circuit([], num_sites=total_qubits)
    for boundary_idx in range(num_clusters - 1):
        left_qubit = (boundary_idx + 1) * cluster_size - 1
        right_qubit = (boundary_idx + 1) * cluster_size
        boundary_qubits.extend([left_qubit, right_qubit])
        bell_circuit = bell_circuit + bell_basis_circuit(
            left_qubit, right_qubit, total_qubits
        )

    prep_and_bell = prep_circuit + bell_circuit
    round_0 = DynamicRound(
        meas_sites=[],
        decoder=lambda _: prep_and_bell,
    )
    round_1 = DynamicRound(
        meas_sites=boundary_qubits,
        decoder=build_ghz_decoder(
            cluster_size, num_clusters, total_qubits, use_cancellations
        ),
    )
    return DynamicCircuit(rounds=[round_0, round_1])
