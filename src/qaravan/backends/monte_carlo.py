"""Monte Carlo backend — Walkers(State) and MonteCarloSimulator(Simulator)."""

from __future__ import annotations
from collections.abc import Iterator
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


class Walkers(State):
    """Equal-weight statistical ensemble of states."""

    def __init__(
        self,
        walkers: State | list[State],
        *,
        num_walkers: int | None = None,
    ):
        if isinstance(walkers, list):
            if not walkers:
                raise ValueError("Walkers requires at least one state.")
            if num_walkers is not None:
                raise ValueError("num_walkers cannot be provided with a list.")
            self._walkers = list(walkers)
            return

        count = 1 if num_walkers is None else num_walkers
        if count <= 0:
            raise ValueError("num_walkers must be positive.")
        self._walkers = [walkers] * count

    @property
    def num_sites(self) -> int:
        return self._walkers[0].num_sites

    @property
    def num_walkers(self) -> int:
        return len(self._walkers)

    @property
    def default_simulator(self) -> type[Simulator]:
        return MonteCarloSimulator

    def __iter__(self) -> Iterator[State]:
        return iter(self._walkers)

    def __getitem__(self, index: int) -> State:
        return self._walkers[index]

    def expectation(self, observable: Observable) -> complex:
        """Average ⟨ψ_i|O|ψ_i⟩ over all walkers."""
        return complex(
            np.mean([walker.expectation(observable) for walker in self._walkers])
        )

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
    """Evolve each walker through independent noise using its member backend."""

    def __init__(
        self,
        circuit: Circuit,
        initial_state: Walkers,
        noise_model: NoiseModel | None = None,
        rng: np.random.Generator | int | None = None,
        decompose: bool = False,
        **member_simulator_kwargs,
    ):
        super().__init__(circuit, initial_state, noise_model, decompose)
        representative = initial_state[0]
        member_simulator_type = representative.default_simulator
        self.member_simulator = member_simulator_type(
            circuit,
            representative,
            noise_model=noise_model,
            rng=rng,
            decompose=decompose,
            **member_simulator_kwargs,
        )

    def _validate_state(self, state: State) -> None:
        if not isinstance(state, Walkers):
            raise IncompatibleStateError(
                f"MonteCarloSimulator requires Walkers; got {type(state).__name__}"
            )

    def run(self) -> Walkers:
        prepared_circuit = self.member_simulator.prepare_circuit()
        final_walkers = []
        for walker in self.initial_state:
            realized_circuit = self._insert_noise(
                prepared_circuit, self.noise_model
            )
            final_walkers.append(self.evolve(walker, realized_circuit))
        return Walkers(final_walkers)

    def _insert_noise(
        self,
        circuit: Circuit,
        noise_model: NoiseModel | None,
    ) -> Circuit:
        return self.member_simulator._insert_noise(circuit, noise_model)

    def translate_gate(self, gate: Gate):
        return self.member_simulator.translate_gate(gate)

    def _apply_translated_gate(self, state: State, translated_gate) -> None:
        self.member_simulator._apply_translated_gate(state, translated_gate)
