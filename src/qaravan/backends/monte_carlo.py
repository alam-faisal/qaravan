"""Monte Carlo backend — Walkers(State) and MonteCarloSimulator(Simulator)."""

from __future__ import annotations
import numpy as np
from qaravan.backends.statevector import Statevector, op_action
from qaravan.core.base import (
    Gate,
    IncompatibleStateError,
    NoiseModel,
    Observable,
    Simulator,
    State,
)
from qaravan.core.circuit import Circuit
from qaravan.core.gates import pauli_string_to_gates


class Walkers(State):
    """Statistical ensemble of Statevectors representing a mixed state."""

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
        return complex(np.mean([sv.expectation(observable) for sv in self._walkers]))

    def sample(
        self, num_shots: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample bitstrings from the ensemble.

        For each shot: pick a random walker, draw one bitstring from its Born distribution.
        Returns (num_shots, num_sites) int array.
        """
        rng = np.random.default_rng(rng)
        walker_indices = rng.integers(0, self.num_walkers, size=num_shots)
        rows = []
        for idx in walker_indices:
            rows.append(self._walkers[idx].sample(1, rng))
        return np.vstack(rows)

    def rdm(self, sites: list[int]) -> np.ndarray:
        """Average reduced density matrix (1/N) Σ_i ρ_i[sites]."""
        return np.mean([sv.rdm(sites) for sv in self._walkers], axis=0)

    def measure_each(
        self, sites: list[int], rng: np.random.Generator | None = None
    ) -> tuple[Walkers, list[str]]:
        """Measure every walker independently; outcomes align with walkers by index."""
        rng = np.random.default_rng(rng)
        collapsed, outcomes = [], []
        for walker in self._walkers:
            state, outcome = walker.measure_and_collapse(sites, rng)
            collapsed.append(state)
            outcomes.append(outcome)
        return Walkers(collapsed), outcomes

    def measure_and_collapse(
        self, sites: list[int], rng: np.random.Generator | None = None
    ) -> tuple[Walkers, str]:
        """Measure the mixed state and posterior-resample an equal-weight ensemble."""
        rng = np.random.default_rng(rng)
        sorted_sites = sorted(sites)

        walker_probabilities = []
        for walker in self._walkers:
            walker_probabilities.append(np.real(np.diag(walker.rdm(sorted_sites))))
        walker_probabilities = np.stack(walker_probabilities)

        mixture_probabilities = walker_probabilities.mean(axis=0)
        mixture_probabilities /= mixture_probabilities.sum()
        outcome_index = rng.choice(len(mixture_probabilities), p=mixture_probabilities)
        outcome = format(outcome_index, f"0{len(sorted_sites)}b")

        posterior_probabilities = walker_probabilities[:, outcome_index]
        posterior_probabilities /= posterior_probabilities.sum()
        ancestor_indices = rng.choice(
            self.num_walkers,
            size=self.num_walkers,
            replace=True,
            p=posterior_probabilities,
        )
        collapsed = []
        for ancestor_index in ancestor_indices:
            collapsed.append(
                self._walkers[ancestor_index].project_and_renorm(sorted_sites, outcome)
            )
        return Walkers(collapsed), outcome

    def __repr__(self) -> str:
        return f"Walkers(num_walkers={self.num_walkers}, num_sites={self.num_sites})"


class MonteCarloSimulator(Simulator):
    """Trajectory-based noisy simulator.

    Runs num_trajectories independent copies of the circuit, inserting sampled
    Pauli noise. Returns a Walkers ensemble of final statevectors.

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
        rng: np.random.Generator | int | None = None,
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
        """Append a freshly sampled noise layer after each layer, or after the whole circuit."""
        if nm is None:
            return circuit
        realized = circuit.copy()
        assert circuit.layers is not None
        if nm.placement == "circuit":
            realized.layers = list(circuit.layers)
            noise_gates = pauli_string_to_gates(nm.sample_error(self._rng))
            if noise_gates:
                realized.layers.append(noise_gates)
            return realized
        noisy_layers = []
        for layer in circuit.layers:
            noisy_layers.append(layer)
            noise_gates = pauli_string_to_gates(nm.sample_error(self._rng))
            if noise_gates:
                noisy_layers.append(noise_gates)
        realized.layers = noisy_layers
        return realized

    def run(self) -> Statevector:
        """One noisy trajectory. Noise is resampled per layer and the initial state is
        deep-copied, so repeated calls are independent."""
        return super().run()

    def run_ensemble(self, num_trajectories: int | None = None) -> Walkers:
        """Independent trajectories, bundled. Defaults to the constructor's num_trajectories."""
        count = self.num_trajectories if num_trajectories is None else num_trajectories
        return Walkers([self.run() for _ in range(count)])

    def translate_gate(self, gate: Gate) -> tuple[np.ndarray, list[int]]:
        return gate.matrix, gate.indices

    def _apply_translated_gate(
        self, state: Statevector, translated_gate: tuple[np.ndarray, list[int]]
    ) -> None:
        mat, indices = translated_gate
        state._tensor = op_action(mat, indices, state._tensor, state.local_dim)
