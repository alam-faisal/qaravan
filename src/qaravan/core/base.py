"""Core abstractions: Gate, State, Simulator, Observable, NoiseModel."""

from __future__ import annotations
import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from qaravan.core.circuits import Circuit


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class IncompatibleNoiseError(Exception):
    pass


class IncompatibleStateError(TypeError):
    pass


# ---------------------------------------------------------------------------
# Gate (ABC)
# ---------------------------------------------------------------------------


class Gate(ABC):
    """Named quantum operation with site indices.

    time: gate duration; only meaningful for noise-aware backends.
    MatrixGate and ParametricGate (concrete subclasses) live in core/gates.py.
    """

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
        """d^k × d^k unitary matrix."""
        ...

    @property
    def local_dim(self) -> int:
        return round(self.matrix.shape[0] ** (1 / self.num_sites))

    @abstractmethod
    def dagger(self) -> Gate: ...

    def __str__(self) -> str:
        return f"{self.name}({self.indices})"

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# State (ABC)
# ---------------------------------------------------------------------------


class State(ABC):
    """Evolving data structure owned by a backend."""

    @property
    @abstractmethod
    def default_simulator(self) -> type[Simulator]:
        """Simulator class used by apply(). Each concrete State subclass sets this."""
        ...

    @abstractmethod
    def expectation(self, observable: Observable) -> complex:
        """Expected value of observable in this state."""
        ...

    @abstractmethod
    def sample(self, num_shots: int) -> np.ndarray:
        """Sample measurement outcomes; returns (num_shots, num_sites) array."""
        ...

    @abstractmethod
    def measure_and_collapse(self, sites: list[int]) -> tuple[State, str]:
        """Measure sites, collapse state; returns (post-measurement State, outcome_str).
        Returned State has the same number of sites as the original."""
        ...

    def partial_overlap(self, other: State, skip: list[int]) -> np.ndarray:
        """Partial ⟨self|other⟩ keeping sites in skip uncontracted.

        Returns (local_dim^|skip| × local_dim^|skip|) matrix.
        skip=[] returns (1,1) full overlap; result[i,j] = ⟨other[j-block]|self[i-block]⟩.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement partial_overlap"
        )

    def overlap(self, other: State) -> complex:
        """⟨self|other⟩. Concrete default via partial_overlap; backends need not override."""
        # partial_overlap[0,0] = ⟨other|self⟩; conjugate to get ⟨self|other⟩
        return self.partial_overlap(other, skip=[])[0, 0].conj()

    @abstractmethod
    def __repr__(self) -> str: ...

    def __str__(self) -> str:
        return repr(self)

    def apply(self, circuit: Circuit, **kwargs) -> State:
        """Evolve self under circuit using the default simulator."""
        return self.default_simulator(circuit, self, **kwargs).run()


# ---------------------------------------------------------------------------
# Observable (ABC)
# ---------------------------------------------------------------------------


class Observable(ABC):
    """Descriptor for a measurable quantity: name, site indices, and local matrix.

    Backends build their preferred representation from obs.matrix and obs.indices.
    State.expectation() dispatches on the concrete subclass — same pattern as
    Simulator dispatching on Gate.
    """

    def __init__(self, name: str, indices: int | list[int]):
        self.name = name
        self.indices = [indices] if isinstance(indices, int) else list(indices)

    @property
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Hermitian matrix for this observable (local or full, depending on subclass)."""
        ...

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
        kwargs can include gate-dependence say for thermal noise,"""
        raise NotImplementedError("subclass must implement `get_kraus`")

    def get_superop(self, *args, **kwargs) -> np.ndarray:
        """Superoperator sum_k (K_k* ⊗ K_k). Works once get_kraus is implemented.

        Convention: K^* ⊗ K is correct for column-major (Fortran-order) vectorization
        of rho, i.e. rho.flatten(order='F'). The DM backend must vectorize consistently.
        """
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
