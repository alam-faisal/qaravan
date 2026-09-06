"""Builders for DynamicCircuit instances for concrete DQC protocols."""

from __future__ import annotations
from typing import Callable
from qaravan.applications.circuit_library import (
    bell_basis_circuit,
    ghz_cluster_prep_circuit,
)
from qaravan.core.base import Gate
from qaravan.core.circuit import Circuit
from qaravan.core.dynamic_circuit import DynamicCircuit, DynamicRound
from qaravan.core.gates import CRX, RX, RZZ, XY, X, Z


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


# ---------------------------------------------------------------------------
# 1D collisional models
#
# M cycles of U = e^{-i H_SA} e^{-i H_S}, then measure every ancilla and reset it to |0>.
# H_SA = (1/2) sum_{(i,I) in B} (X_i X_I + Y_i Y_I) couples each system qubit to its ancilla.
# ---------------------------------------------------------------------------


def interleaved_collisional_layout(
    num_system: int,
) -> tuple[list[int], list[int], list[tuple[int, int]]]:
    """One ancilla per system qubit, interleaved: system 2k, ancilla 2k+1.

    Returns (system_sites, ancilla_sites, coupling_pairs); interleaving keeps every
    system-ancilla pair adjacent.
    """
    system_sites = [2 * k for k in range(num_system)]
    ancilla_sites = [2 * k + 1 for k in range(num_system)]
    coupling_pairs = list(zip(system_sites, ancilla_sites))
    return system_sites, ancilla_sites, coupling_pairs


def dissipative_ising_layer(
    system_sites: list[int], omega: float, field_v: float, dt: float = 1.0
) -> list[Gate]:
    """First-order Trotter step of H_S = omega sum_i X_i + field_v sum_i Z_i Z_{i+1}.

    Ordering e^{-i H_ZZ} e^{-i H_X}, with the ZZ bonds split even-then-odd.
    """
    gates: list[Gate] = []
    if omega:
        gates.extend(RX(site, omega * dt) for site in system_sites)
    if field_v:
        bonds = list(zip(system_sites, system_sites[1:]))
        for parity in (0, 1):
            gates.extend(
                RZZ([left, right], field_v * dt) for left, right in bonds[parity::2]
            )
    return gates


def qcp_layer(system_sites: list[int], omega: float, dt: float = 1.0) -> list[Gate]:
    """First-order Trotter step of the quantum contact process,
    H_S = (omega/2) sum_i (X_i n_{i+1} + n_i X_{i+1}), bonds split even-then-odd.

    Each bond factorises as CRX(control, target) in both directions; the half-angle in H_S
    and CRX's full-angle convention cancel, so the gate angle is omega*dt/2.
    """
    bonds = list(zip(system_sites, system_sites[1:]))
    theta = omega * dt / 2
    gates: list[Gate] = []
    for parity in (0, 1):
        for left, right in bonds[parity::2]:
            gates.append(CRX([right, left], theta))
            gates.append(CRX([left, right], theta))
    return gates


def collisional_dqc(
    system_layer: list[Gate],
    ancilla_sites: list[int],
    coupling_pairs: list[tuple[int, int]],
    phi: float,
    num_cycles: int,
    num_sites: int,
) -> DynamicCircuit:
    """M cycles of (evolve, measure every ancilla, reset it to |0>).

    phi is the system-ancilla coupling angle, so a system qubit in |1> decays with
    probability sin^2(phi) per cycle. For a target Lindblad rate, phi = sqrt(rate * dt).

    system_layer is the already-built H_S Trotter step, which is what lets one protocol serve
    both the dissipative Ising model and the QCP. run() returns num_cycles outcomes, one per
    cycle; the fraction of '1's in each is that cycle's activity. Every cycle ends with the
    reset, the last one included, so the returned state always has its ancillas in |0>.
    """
    coupling_gates = [
        XY([system, ancilla], phi / 2) for system, ancilla in coupling_pairs
    ]
    evolution = Circuit(list(system_layer) + coupling_gates, num_sites=num_sites)

    def reset(outcome: str) -> list[Gate]:
        return [X(site) for site, bit in zip(ancilla_sites, outcome) if bit == "1"]

    def reset_then_evolve(outcome: str) -> Circuit:
        return Circuit(reset(outcome) + evolution.gates, num_sites=num_sites)

    rounds = [DynamicRound(decoder=lambda _: evolution)]
    rounds.extend(
        DynamicRound(meas_sites=list(ancilla_sites), decoder=reset_then_evolve)
        for _ in range(num_cycles - 1)
    )
    rounds.append(
        DynamicRound(
            meas_sites=list(ancilla_sites),
            decoder=lambda outcome: Circuit(reset(outcome), num_sites=num_sites),
        )
    )
    return DynamicCircuit(rounds=rounds)
