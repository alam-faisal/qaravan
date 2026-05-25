"""Circuit class, gate-embedding helpers, and circuit generation functions."""

from __future__ import annotations

import numpy as np

from qaravan.core.base import Gate
from qaravan.core.gates import CNOT, H, ParametricGate, RX, RY, RZ, RXX, RYY, RZZ


class Circuit:
    """An ordered sequence of Gates."""

    def __init__(
        self,
        gates: list[Gate],
        num_sites: int | None = None,
        local_dim: int = 2,
    ):
        self.gates = list(gates)
        self.local_dim = local_dim
        self.layers: list[list[Gate]] | None = None

        if num_sites is not None:
            self.num_sites = num_sites
        elif gates:
            self.num_sites = max(idx for g in gates for idx in g.indices) + 1
        else:
            self.num_sites = 0

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
            num_sites=self.num_sites,
            local_dim=self.local_dim,
        )

    def to_matrix(self) -> np.ndarray:
        """Full unitary matrix of the circuit. For debugging; real backends never build this."""
        dim = self.local_dim**self.num_sites
        result = np.eye(dim, dtype=complex)
        for gate in self.gates:
            full = _embed_gate(gate, self.num_sites, self.local_dim)
            result = full @ result
        return result

    @property
    def num_params(self) -> int:
        """Total number of continuous parameters across all ParametricGates."""
        return sum(g.num_params for g in self.gates if isinstance(g, ParametricGate))

    def bind(self, params: np.ndarray | list[float]) -> Circuit:
        """Return a new Circuit with ParametricGate parameters replaced by params.

        Parameters are consumed sequentially in gate order.
        """
        assert len(params) == self.num_params, (
            f"Expected {self.num_params} params, got {len(params)}"
        )
        new_gates: list[Gate] = []
        idx = 0
        for gate in self.gates:
            if isinstance(gate, ParametricGate):
                new_gates.append(
                    type(gate)(gate.indices, *params[idx : idx + gate.num_params], gate.time)
                )
                idx += gate.num_params
            else:
                new_gates.append(gate)
        return Circuit(new_gates, num_sites=self.num_sites, local_dim=self.local_dim)

    def copy(self) -> Circuit:
        """Shallow copy of gate list; layers reset to None."""
        return Circuit(
            list(self.gates), num_sites=self.num_sites, local_dim=self.local_dim
        )

    def __add__(self, other: Circuit) -> Circuit:
        if not isinstance(other, Circuit):
            return NotImplemented
        return Circuit(
            self.gates + other.gates,
            num_sites=max(self.num_sites, other.num_sites),
            local_dim=self.local_dim,
        )

    def __mul__(self, n: int) -> Circuit:
        if not isinstance(n, int) or n < 0:
            return NotImplemented
        return Circuit(
            self.gates * n, num_sites=self.num_sites, local_dim=self.local_dim
        )

    def __rmul__(self, n: int) -> Circuit:
        return self.__mul__(n)

    def __len__(self) -> int:
        return len(self.gates)

    def __getitem__(self, key: int | slice) -> Gate | Circuit:
        if isinstance(key, int):
            return self.gates[key]
        return Circuit(
            self.gates[key], num_sites=self.num_sites, local_dim=self.local_dim
        )

    def __str__(self) -> str:
        return f"Circuit(num_sites={self.num_sites}, gates={self.gates})"

    def __repr__(self) -> str:
        return str(self)


def _embed_gate(gate: Gate, num_sites: int, local_dim: int) -> np.ndarray:
    """Embed gate matrix into the full num_sites-site Hilbert space.

    For debugging and to_matrix(); real backends apply gates without building full matrices.
    """
    k = gate.num_sites
    sorted_indices = sorted(gate.indices)

    if sorted_indices == list(range(sorted_indices[0], sorted_indices[0] + k)):
        left_dim = local_dim ** sorted_indices[0]
        right_dim = local_dim ** (num_sites - sorted_indices[0] - k)
        return np.kron(np.kron(np.eye(left_dim), gate.matrix), np.eye(right_dim))

    # non-contiguous sites: permute basis, apply gate on leading sites, permute back
    dim = local_dim**num_sites
    perm = sorted_indices + [i for i in range(num_sites) if i not in sorted_indices]
    perm_mat = np.zeros((dim, dim))
    for i in range(dim):
        digits = _int_to_digits(i, local_dim, num_sites)
        j = _digits_to_int([digits[p] for p in perm], local_dim)
        perm_mat[i, j] = 1.0
    gate_full = np.kron(gate.matrix, np.eye(local_dim ** (num_sites - k)))
    return perm_mat.T @ gate_full @ perm_mat


# ---------------------------------------------------------------------------
# Circuit generation functions
# ---------------------------------------------------------------------------


def nn_pairs(n: int, periodic: bool = False) -> list[list[int]]:
    """Nearest-neighbor index pairs for a 1D chain of n sites.

    Returns [[0,1],[1,2],...,[n-2,n-1]], plus [n-1,0] if periodic=True.
    """
    pairs = [[i, i + 1] for i in range(n - 1)]
    if periodic and n > 1:
        pairs.append([n - 1, 0])
    return pairs


def ghz_circuit(n: int) -> Circuit:
    """H(0) followed by CNOT(i, i+1) for i in 0..n-2."""
    gates: list[Gate] = [H(0)] + [CNOT([i, i + 1]) for i in range(n - 1)]
    return Circuit(gates, num_sites=n)


def rx_layer(
    n: int,
    params: np.ndarray | list[float] | None = None,
    seed: int | None = None,
) -> Circuit:
    """Circuit of n RX gates, one per site. Random params in [0, 2π) if not provided."""
    if params is None:
        params = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    return Circuit([RX(i, params[i]) for i in range(n)])


def ry_layer(
    n: int,
    params: np.ndarray | list[float] | None = None,
    seed: int | None = None,
) -> Circuit:
    """Circuit of n RY gates, one per site. Random params in [0, 2π) if not provided."""
    if params is None:
        params = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    return Circuit([RY(i, params[i]) for i in range(n)])


def rz_layer(
    n: int,
    params: np.ndarray | list[float] | None = None,
    seed: int | None = None,
) -> Circuit:
    """Circuit of n RZ gates, one per site. Random params in [0, 2π) if not provided."""
    if params is None:
        params = np.random.default_rng(seed).uniform(0, 2 * np.pi, n)
    return Circuit([RZ(i, params[i]) for i in range(n)])


def rxx_layer(
    skeleton: list[list[int]],
    params: np.ndarray | list[float] | None = None,
    seed: int | None = None,
) -> Circuit:
    """RXX gates on each pair in skeleton. Random params in [0, 2π) if not provided."""
    if params is None:
        params = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(skeleton))
    return Circuit([RXX(skeleton[i], params[i]) for i in range(len(skeleton))])


def ryy_layer(
    skeleton: list[list[int]],
    params: np.ndarray | list[float] | None = None,
    seed: int | None = None,
) -> Circuit:
    """RYY gates on each pair in skeleton. Random params in [0, 2π) if not provided."""
    if params is None:
        params = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(skeleton))
    return Circuit([RYY(skeleton[i], params[i]) for i in range(len(skeleton))])


def rzz_layer(
    skeleton: list[list[int]],
    params: np.ndarray | list[float] | None = None,
    seed: int | None = None,
) -> Circuit:
    """RZZ gates on each pair in skeleton. Random params in [0, 2π) if not provided."""
    if params is None:
        params = np.random.default_rng(seed).uniform(0, 2 * np.pi, len(skeleton))
    return Circuit([RZZ(skeleton[i], params[i]) for i in range(len(skeleton))])


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
