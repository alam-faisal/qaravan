"""Core abstractions: Gate, State, Simulator, Observable, NoiseModel.

Circuit lives in core.circuits and is re-exported here for convenience.
"""
from __future__ import annotations
import copy
from abc import ABC, abstractmethod
import numpy as np


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class IncompatibleNoiseError(Exception):
    pass

class IncompatibleStateError(TypeError):
    pass


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

class Gate:
    """A named operation with site indices and a matrix representation.

    time: gate duration, usually set at compile time by Simulator; only
    meaningful for noise-aware backends.
    """

    def __init__(
        self,
        name: str,
        indices: int | list[int],
        matrix: np.ndarray,
        time: float | None = None,
    ):
        self.name = name
        self.indices = [indices] if isinstance(indices, int) else list(indices)
        self.matrix = np.asarray(matrix, dtype=complex)
        self.time = time
        self.local_dim = round(self.matrix.shape[0] ** (1 / self.num_sites))

    @property
    def num_sites(self) -> int:
        return len(self.indices)

    def dagger(self) -> Gate:
        return Gate(
            self.name + "†",
            self.indices,
            self.matrix.conj().T,
            self.time,
        )

    def __str__(self) -> str:
        return f"{self.name}({self.indices})"

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# State (ABC)
# ---------------------------------------------------------------------------

class State(ABC):
    """Evolving data structure owned by a backend."""

    @abstractmethod
    def expectation(self, observable: Observable) -> complex:
        """Expected value of observable in this state."""
        ...

    @abstractmethod
    def sample(self, num_shots: int) -> np.ndarray:
        """Sample measurement outcomes; returns (num_shots, num_sites) array."""
        ...

    @abstractmethod
    def sample_and_collapse(self, sites: list[int]) -> tuple[str, State]:
        """Measure sites, return (outcome_string, post-measurement state)."""
        ...

    @abstractmethod
    def overlap(self, other: State) -> complex:
        """⟨self|other⟩ (or tr(self† other) for mixed states)."""
        ...


# ---------------------------------------------------------------------------
# Observable (ABC)
# ---------------------------------------------------------------------------

class Observable(ABC):
    """Descriptor for a measurable quantity: name and site indices.

    No matrix lives here. State.expectation() dispatches on the concrete
    subclass and builds whatever representation it needs — same pattern as
    Simulator dispatching on Gate.
    """

    def __init__(self, name: str, indices: int | list[int]):
        self.name = name
        self.indices = [indices] if isinstance(indices, int) else list(indices)

    def __str__(self) -> str:
        return f"{self.name}({self.indices})"


# ---------------------------------------------------------------------------
# NoiseModel (ABC)
# ---------------------------------------------------------------------------

class NoiseModel(ABC):
    """Base noise model. Subclasses implement get_kraus and optionally sample_error."""
    @property
    @abstractmethod
    def gate_dependent(self) -> bool: ...

    def get_kraus(self, *args, **kwargs) -> list[np.ndarray]:
        """Kraus operators for the noise channel following gate.
        kwargs can include gate-dependence say for thermal noise, """
        raise NotImplementedError("subclass must implement `get_kraus`")

    def get_superop(self, *args, **kwargs) -> np.ndarray:
        """Superoperator sum_k (K_k* ⊗ K_k). Works once get_kraus is implemented.
        Again kwargs can include gate-dependence."""
        kraus = self.get_kraus(*args, **kwargs)
        return sum(np.kron(k.conj(), k) for k in kraus)

    def sample_error(self) -> np.ndarray:
        """Sample a Pauli error. Not all noise models support this."""
        raise NotImplementedError("subclass must implement `sample_error`")


# ---------------------------------------------------------------------------
# Simulator (ABC)
# ---------------------------------------------------------------------------

class Simulator(ABC):
    """Takes (circuit, initial_state, noise_model), compiles, evolves, returns State."""

    def __init__(
        self,
        circuit: Circuit,
        initial_state: State,
        noise_model: NoiseModel | None = None,
        decompose: bool = False,
    ):
        self._validate_state(initial_state)
        self.circuit = circuit
        self.initial_state = initial_state
        self.noise_model = noise_model
        self.decompose = decompose

    def run(self) -> State:
        """Compile: (decompose →) construct_layers → _insert_noise → evolve."""
        state = copy.deepcopy(self.initial_state)
        circuit = self.circuit.copy()
        if self.decompose:
            circuit.decompose()
        circuit.construct_layers()
        circuit = self._insert_noise(circuit, self.noise_model)
        for layer in circuit.layers:
            for gate in layer:
                translated = self.translate_gate(gate)
                self._apply_translated_gate(state, translated)
        return state

    @abstractmethod
    def _validate_state(self, state: State) -> None:
        """Raise IncompatibleStateError if state type is wrong for this backend."""
        ...

    @abstractmethod
    def translate_gate(self, gate: Gate):
        """Convert gate to backend-specific representation."""
        ...

    @abstractmethod
    def _apply_translated_gate(self, state: State, translated_gate) -> None:
        """Apply a translated gate to state in-place."""
        ...

    def _insert_noise(self, circuit: Circuit, nm: NoiseModel | None) -> Circuit:
        """Default: raise if noise model provided. Noise-compatible backends override."""
        if nm is not None:
            raise IncompatibleNoiseError(
                f"{type(self).__name__} does not support noise models; "
                "use DensityMatrixSimulator or MonteCarloSimulator"
            )
        return circuit


# Re-export Circuit here so `from qaravan.core.base import Circuit` keeps working.
# Gate is fully defined above before this import, so the circular reference resolves cleanly.
from qaravan.core.circuits import Circuit  # noqa: E402
