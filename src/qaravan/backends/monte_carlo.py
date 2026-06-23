"""Monte Carlo backend — Walkers(State) and MonteCarloSimulator(Simulator)."""

from __future__ import annotations

import copy

import numpy as np

from qaravan.core.base import (
    Gate,
    IncompatibleStateError,
    NoiseModel,
    Observable,
    Simulator,
    State,
)
from qaravan.core.circuit import Circuit
from qaravan.core.gates import X, Y, Z
from qaravan.backends.statevector import Statevector, op_action

_PAULI_GATE_MAP = {"X": X, "Y": Y, "Z": Z}


def _pauli_string_to_gates(string: str) -> list[Gate]:
    """Convert e.g. 'XIZYI' → [X(0), Z(2), Y(3)]. Identity sites are skipped."""
    gates = []
    for site, char in enumerate(string.upper()):
        if char in _PAULI_GATE_MAP:
            gates.append(_PAULI_GATE_MAP[char](site))
    return gates


class Walkers(State):
    """Statistical ensemble of Statevectors representing a mixed state via trajectories.

    Expectation values and reduced density matrices are computed as averages over walkers.
    measure_and_collapse is not defined — it is handled at the MonteCarloSimulator level.
    """

    def __init__(self, statevectors: list[Statevector]):
        if not statevectors:
            raise ValueError("Walkers requires at least one Statevector.")
        self._walkers = list(statevectors)

    @property
    def num_sites(self) -> int:
        return self._walkers[0].num_sites

    @property
    def num_walkers(self) -> int:
        return len(self._walkers)

    @property
    def default_simulator(self):
        raise NotImplementedError(
            "Walkers cannot be used as an initial state for apply(). "
            "Pass a Statevector to MonteCarloSimulator instead."
        )

    def apply(self, circuit: Circuit, **kwargs):
        raise NotImplementedError(
            "Walkers.apply() is not supported. "
            "Pass a Statevector to MonteCarloSimulator instead."
        )

    def expectation(self, observable: Observable) -> complex:
        """Average ⟨ψ_i|O|ψ_i⟩ over all walkers."""
        return complex(
            np.mean([sv.expectation(observable) for sv in self._walkers])
        )

    def sample(self, num_shots: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample bitstrings from the ensemble.

        For each shot: pick a random walker, draw one bitstring from its Born distribution.
        Returns (num_shots, num_sites) int array.
        """
        rng = np.random.default_rng(rng)
        walker_indices = rng.integers(0, self.num_walkers, size=num_shots)
        rows = []
        for idx in walker_indices:
            rows.append(self._walkers[idx].sample(1))
        return np.vstack(rows)

    def rdm(self, sites: list[int]) -> np.ndarray:
        """Average reduced density matrix (1/N) Σ_i ρ_i[sites]."""
        return np.mean([sv.rdm(sites) for sv in self._walkers], axis=0)

    def measure_and_collapse(self, sites: list[int]) -> tuple[State, str]:
        raise NotImplementedError(
            "Walkers does not support measure_and_collapse. "
            "Mid-circuit measurements on a trajectory ensemble are handled by "
            "MonteCarloSimulator, which collapses each walker independently."
        )

    def __repr__(self) -> str:
        return f"Walkers(num_walkers={self.num_walkers}, num_sites={self.num_sites})"


class MonteCarloSimulator(Simulator):
    """Trajectory-based noisy simulator.

    Runs num_trajectories independent copies of the circuit, inserting sampled
    Pauli noise after each layer. Returns a Walkers ensemble of final statevectors.

    noise_model must implement sample_error(rng) -> str (Pauli string). Only PauliNoise
    and DepolarizingNoise are supported; ThermalNoise will raise at runtime.
    rng: seed for reproducibility; pass None for random.
    """

    def __init__(
        self,
        circuit: Circuit,
        initial_state: Statevector,
        noise_model: NoiseModel | None = None,
        num_trajectories: int = 500,
        rng: np.random.Generator | None = None,
        decompose: bool = False,
    ):
        super().__init__(circuit, initial_state, noise_model, decompose)
        self.num_trajectories = num_trajectories
        self._rng = np.random.default_rng(rng)

    def _validate_state(self, state: State) -> None:
        if not isinstance(state, Statevector):
            raise IncompatibleStateError(
                f"MonteCarloSimulator requires a Statevector; got {type(state).__name__}"
            )

    def _insert_noise(self, circuit: Circuit, nm: NoiseModel | None) -> Circuit:
        """Interleave a freshly sampled noise layer after each circuit layer."""
        if nm is None:
            return circuit
        noisy_layers = []
        for layer in circuit.layers:
            noisy_layers.append(layer)
            noise_gates = _pauli_string_to_gates(nm.sample_error(self._rng))
            if noise_gates:
                noisy_layers.append(noise_gates)
        circuit.layers = noisy_layers
        return circuit

    def run(self) -> Walkers:
        """Run num_trajectories independent trajectories and return a Walkers ensemble."""
        _run_one = super().run
        walkers = [_run_one() for _ in range(self.num_trajectories)]
        return Walkers(walkers)

    def translate_gate(self, gate: Gate) -> tuple[np.ndarray, list[int]]:
        return gate.matrix, gate.indices

    def _apply_translated_gate(
        self, state: Statevector, translated_gate: tuple[np.ndarray, list[int]]
    ) -> None:
        mat, indices = translated_gate
        state._tensor = op_action(mat, indices, state._tensor, state.local_dim)
