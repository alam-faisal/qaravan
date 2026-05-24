"""Core abstractions: Gate, Circuit, State, Simulator, Observable, NoiseModel."""
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
# Circuit
# ---------------------------------------------------------------------------

class Circuit:
    """An ordered sequence of Gates."""

    def __init__(
        self,
        gates: list[Gate],
        n: int | None = None,
        local_dim: int = 2,
    ):
        self.gates = list(gates)
        self.local_dim = local_dim
        self.layers: list[list[Gate]] | None = None

        if n is not None:
            self.n = n
        elif gates:
            self.n = max(idx for g in gates for idx in g.indices) + 1
        else:
            self.n = 0

    def construct_layers(self) -> None:
        """Greedy layer packing: assigns each gate to the earliest layer it fits."""
        layers: list[list[Gate]] = []
        occupied: list[set[int]] = []

        for gate in self.gates:
            site_set = set(gate.indices)
            placed = False
            for i, layer_sites in enumerate(occupied):
                if layer_sites.isdisjoint(site_set):
                    layers[i].append(gate)
                    layer_sites.update(site_set)
                    placed = True
                    break
            if not placed:
                layers.append([gate])
                occupied.append(set(site_set))

        self.layers = layers

    def decompose(self, basis: str = "ZSX") -> None:
        """Expand composite gates into basis gates; mutates self.gates."""
        raise NotImplementedError("decompose not yet implemented")

    def dagger(self) -> Circuit:
        """Reversed gate order with each gate conjugate-transposed."""
        return Circuit(
            [g.dagger() for g in reversed(self.gates)],
            n=self.n,
            local_dim=self.local_dim,
        )

    def to_matrix(self) -> np.ndarray:
        """Full unitary matrix of the circuit."""
        dim = self.local_dim ** self.n
        result = np.eye(dim, dtype=complex)
        for gate in self.gates:
            full = _embed_gate(gate, self.n, self.local_dim)
            result = full @ result
        return result

    def copy(self) -> Circuit:
        """Shallow copy of gate list; layers reset to None."""
        return Circuit(list(self.gates), n=self.n, local_dim=self.local_dim)

    def __add__(self, other: Circuit) -> Circuit:
        if not isinstance(other, Circuit):
            return NotImplemented
        return Circuit(self.gates + other.gates, n=max(self.n, other.n), local_dim=self.local_dim)

    def __mul__(self, n: int) -> Circuit:
        if not isinstance(n, int) or n < 0:
            return NotImplemented
        return Circuit(self.gates * n, n=self.n, local_dim=self.local_dim)

    def __rmul__(self, n: int) -> Circuit:
        return self.__mul__(n)

    def __len__(self) -> int:
        return len(self.gates)

    def __getitem__(self, key: int | slice) -> Gate | Circuit:
        if isinstance(key, int):
            return self.gates[key]
        return Circuit(self.gates[key], n=self.n, local_dim=self.local_dim)

    def __str__(self) -> str:
        return f"Circuit(n={self.n}, gates={self.gates})"

    def __repr__(self) -> str:
        return str(self)


def _embed_gate(gate: Gate, n: int, local_dim: int) -> np.ndarray:
    """Embed gate matrix into full n-site Hilbert space via permutation."""
    k = gate.num_sites
    sorted_indices = sorted(gate.indices)

    if sorted_indices == list(range(sorted_indices[0], sorted_indices[0] + k)):
        left_dim = local_dim ** sorted_indices[0]
        right_dim = local_dim ** (n - sorted_indices[0] - k)
        return np.kron(np.kron(np.eye(left_dim), gate.matrix), np.eye(right_dim))

    # non-contiguous sites: permute basis, apply gate on leading sites, permute back
    dim = local_dim ** n
    perm = sorted_indices + [i for i in range(n) if i not in sorted_indices]
    perm_mat = np.zeros((dim, dim))
    for i in range(dim):
        digits = _int_to_digits(i, local_dim, n)
        j = _digits_to_int([digits[p] for p in perm], local_dim)
        perm_mat[i, j] = 1.0
    gate_full = np.kron(gate.matrix, np.eye(local_dim ** (n - k)))
    return perm_mat.T @ gate_full @ perm_mat


def _int_to_digits(i: int, base: int, n: int) -> list[int]:
    digits = []
    for _ in range(n):
        digits.append(i % base)
        i //= base
    return digits[::-1]


def _digits_to_int(digits: list[int], base: int) -> int:
    result = 0
    for d in digits:
        result = result * base + d
    return result


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
    def sample(self, shots: int) -> np.ndarray:
        """Sample measurement outcomes; returns (shots,) or (shots, n) array."""
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
    """Base noise model. Modelled on legacy Noise class."""

    def get_kraus(self, gate: Gate) -> list[np.ndarray]:
        """Kraus operators for the noise channel following gate."""
        raise NotImplementedError("subclass must implement `get_kraus`")

    def get_superop(self, gate: Gate) -> np.ndarray:
        """Superoperator sum_k (K_k* ⊗ K_k). Works once get_kraus is implemented."""
        kraus = self.get_kraus(gate)
        return sum(np.kron(k.conj(), k) for k in kraus)

    def sample_error(self, gate: Gate) -> np.ndarray:
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
