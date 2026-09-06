"""Builders for DynamicCircuit instances for concrete DQC protocols."""

from __future__ import annotations
from typing import Callable
import numpy as np
from qaravan.applications.circuit_library import (
    bell_basis_circuit,
    ghz_cluster_prep_circuit,
)
from qaravan.backends.monte_carlo import MonteCarloSimulator, Walkers
from qaravan.backends.statevector import Statevector
from qaravan.core.base import Gate, NoiseModel, State
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


def reset_decoder(qubits: list[int], num_sites: int) -> Callable[[str], Circuit]:
    """Decoder returning X on each of qubits that measured 1, so all of them end in |0>.

    Returns decoder(outcome) -> Circuit.
    outcome: bitstring of len(qubits),
    """

    def decoder(outcome: str) -> Circuit:
        gates = [X(site) for site, bit in zip(qubits, outcome) if bit == "1"]
        return Circuit(gates, num_sites=num_sites)

    return decoder


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

    system_layer is the already-built H_S Trotter step
    run() returns num_cycles outcomes, one per cycle
    """
    coupling_gates = [
        XY([system, ancilla], phi / 2) for system, ancilla in coupling_pairs
    ]
    evolution = Circuit(list(system_layer) + coupling_gates, num_sites=num_sites)

    reset = reset_decoder(ancilla_sites, num_sites)

    def reset_then_evolve(outcome: str) -> Circuit:
        return Circuit(reset(outcome).gates + evolution.gates, num_sites=num_sites)

    rounds = [DynamicRound(decoder=lambda _: evolution)]
    rounds.extend(
        DynamicRound(meas_sites=list(ancilla_sites), decoder=reset_then_evolve)
        for _ in range(num_cycles - 1)
    )
    rounds.append(DynamicRound(meas_sites=list(ancilla_sites), decoder=reset))
    return DynamicCircuit(rounds=rounds)


def noisy_dqc_trajectories(
    dqc: DynamicCircuit,
    initial_state: Statevector,
    noise_model: NoiseModel,
    num_trajectories: int,
    rng: np.random.Generator | None = None,
    observer: Callable[[int, State], None] | None = None,
) -> tuple[Walkers, list[list[str]]]:
    """Run a DynamicCircuit under sampled Pauli noise, once per trajectory.

    Returns (ensemble of final states, one measurement record per trajectory).

    The ensemble is over noise realisations and measurement outcomes jointly: each trajectory
    is a single state following its own record down its own decoder branch, which is why the
    rounds cannot be run on a Walkers ensemble directly.

    Noise follows gates, so a round whose decoder returns no gates picks up none; idling
    qubits are a duration-aware noise model's job, not this one's.

    Measurement and noise draw from separately spawned streams, so an error_rate of 0 leaves
    the measurement record identical to a noiseless run from the same seed -- sharing one
    stream would desynchronise it, since sample_error draws even when it returns identity.
    """
    measurement_rng, noise_rng = np.random.default_rng(rng).spawn(2)

    def apply_noisily(state: State, circuit: Circuit) -> State:
        return MonteCarloSimulator(circuit, state, noise_model, rng=noise_rng).run()

    finals = []
    records = []
    for _ in range(num_trajectories):
        final, outcomes = dqc.run(
            initial_state, observer=observer, rng=measurement_rng, apply=apply_noisily
        )
        finals.append(final)
        records.append(outcomes)
    return Walkers(finals), records
