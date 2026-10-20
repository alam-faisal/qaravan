"""Core abstractions: Gate, State, Simulator, Observable, NoiseModel."""

from __future__ import annotations
import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from qaravan.core._repr import named

if TYPE_CHECKING:
    from qaravan.core.circuit import Circuit


# === Exceptions ===


class IncompatibleNoiseError(Exception):
    pass


class IncompatibleStateError(TypeError):
    pass


# === Gate ===


class Gate(ABC):
    """Backend-independent gate; time matters only to noise-aware backends."""

    def __init__(
        self,
        name: str,
        indices: int | list[int],
        time: float | None = None,
    ):
        self.name = name
        self.indices = [indices] if isinstance(indices, int) else list(indices)
        self.num_sites = len(self.indices)
        self.time = time

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Local d^k × d^k matrix representation."""
        ...

    @property
    def local_dim(self) -> int:
        return round(self.matrix.shape[0] ** (1 / self.num_sites))

    @abstractmethod
    def dagger(self) -> Gate: ...

    def __repr__(self) -> str:
        return named(self.name, self.indices)


# === State ===


class State(ABC):
    """The data structure being evolved during a simulation."""

    @property
    @abstractmethod
    def default_simulator(self) -> type[Simulator]:
        """Simulator used by apply()."""
        ...

    def apply(self, circuit: Circuit, **kwargs) -> State:
        """Evolve through a circuit using the default simulator."""
        return self.default_simulator(circuit, self, **kwargs).run()

    @abstractmethod
    def expectation(self, observable: Observable) -> complex: ...

    @abstractmethod
    def sample(
        self, num_shots: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample outcomes as a (num_shots, num_sites) array."""
        ...

    @abstractmethod
    def measure_and_collapse(
        self, sites: list[int], rng: np.random.Generator | None = None
    ) -> tuple[State, str]:
        """Return the post-measurement state and outcome bitstring."""
        ...

    def partial_overlap(self, other: State, skip: list[int]) -> np.ndarray:
        """Partial ⟨self|other⟩ with sites in skip left uncontracted."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement partial_overlap"
        )

    def overlap(self, other: State) -> complex:
        """Return ⟨self|other⟩ via partial_overlap."""
        # partial_overlap[0,0] = ⟨other|self⟩; conjugate to get ⟨self|other⟩
        return self.partial_overlap(other, skip=[])[0, 0].conj()

    @abstractmethod
    def __repr__(self) -> str: ...


# === Observable ===


class Observable(ABC):
    """Backend-independent observable."""

    def __init__(self, name: str, indices: int | list[int]):
        self.name = name
        self.indices = [indices] if isinstance(indices, int) else list(indices)
        self.span = len(self.indices)

    @abstractmethod
    def to_matrix(self) -> np.ndarray:
        """Dense matrix acting on the observable's indexed sites."""
        ...

    def __repr__(self) -> str:
        return named(self.name, self.indices)


# === NoiseModel ===


class NoiseModel(ABC):
    """Noise channel exposed through exact or sampled representations."""

    @property
    @abstractmethod
    def gate_dependent(self) -> bool: ...

    def get_kraus(self, *args, **kwargs) -> list[np.ndarray]:
        """Kraus operators for the channel following a gate."""
        raise NotImplementedError("subclass must implement `get_kraus`")

    def get_superop(self, *args, **kwargs) -> np.ndarray:
        """Return Σ_k K_k* ⊗ K_k for column-major vectorization."""
        kraus = self.get_kraus(*args, **kwargs)
        return sum(np.kron(k.conj(), k) for k in kraus)

    def sample_error(
        self, rng: np.random.Generator | int | None = None
    ) -> str:
        """Sample a Pauli error."""
        raise NotImplementedError("subclass must implement `sample_error`")

    def implements(self, method_name: str) -> bool:
        """Whether this subclass overrides the NoiseModel method `method_name`."""
        return getattr(type(self), method_name) is not getattr(NoiseModel, method_name)


# === Simulator ===


class Simulator(ABC):
    """Takes (circuit, initial_state, noise_model), compiles, evolves, returns State."""

    def __init__(
        self,
        circuit: Circuit,
        initial_state: State,
        noise_model: NoiseModel | None = None,
        decompose: bool = False,
        *,
        noise_rng: np.random.Generator | int | None = None,
    ):
        self._validate_state(initial_state)
        if noise_model is not None:
            self._validate_noise(noise_model)
        self.circuit = circuit
        self.initial_state = initial_state
        self.noise_model = noise_model
        self.decompose = decompose
        self.noise_rng = np.random.default_rng(noise_rng)

    def run(self) -> State:
        """Prepare the circuit, insert its noise, and evolve the initial state."""
        prepared_circuit = self._prepare_circuit()
        realized_circuit = self._insert_noise(prepared_circuit, self.noise_model)
        return self._evolve(self.initial_state, realized_circuit)

    def _prepare_circuit(self) -> Circuit:
        """Copy, optionally decompose, and layer the circuit."""
        circuit = self.circuit.copy()
        if self.decompose:
            circuit.decompose()
        circuit.construct_layers()
        return circuit

    def _evolve(self, initial_state: State, prepared_circuit: Circuit) -> State:
        """Apply a prepared circuit to a copy of initial_state."""
        state = copy.deepcopy(initial_state)
        assert prepared_circuit.layers is not None
        for layer in prepared_circuit.layers:
            for gate in layer:
                translated = self.translate_gate(gate)
                self._apply_translated_gate(state, translated)
        return state

    @abstractmethod
    def _insert_noise(
        self, circuit: Circuit, noise_model: NoiseModel | None
    ) -> Circuit:
        """Return the circuit with noise realized; noise_model has passed _validate_noise."""
        ...

    @abstractmethod
    def translate_gate(self, gate: Gate):
        """Convert gate to backend-specific representation."""
        ...

    @abstractmethod
    def _apply_translated_gate(self, state: State, translated_gate) -> None:
        """Apply a translated gate to state in-place."""
        ...

    @abstractmethod
    def _validate_noise(self, noise_model: NoiseModel) -> None:
        """Raise IncompatibleNoiseError if this backend cannot realize noise_model."""
        ...

    @abstractmethod
    def _validate_state(self, state: State) -> None:
        """Reject state types unsupported by this backend."""
        ...