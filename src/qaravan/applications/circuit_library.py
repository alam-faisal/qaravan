"""Common circuit instances and parametric layer factories."""

from __future__ import annotations

import numpy as np

from qaravan.core.circuit import Circuit
from qaravan.core.gates import (
    CNOT,
    H,
    Gate,
    MatrixGate,
    RX,
    RY,
    RZ,
    RXX,
    RYY,
    RZZ,
    random_unitary,
)


def brickwall_skeleton(Lx: int, Ly: int = 1) -> list[list[int]]:
    """NN edges in brickwall order for Lx×Ly grid (row-major, 0-indexed).

    One pass = even-x horiz, odd-x horiz, even-y vert, odd-y vert.
    For depth-d circuits: brickwall_skeleton(Lx, Ly) * d.
    For 1D (Ly=1) gives the standard alternating brickwall.
    """
    h_edges: list[tuple[int, int]] = []
    v_edges: list[tuple[int, int]] = []
    for y in range(Ly):
        for x in range(Lx):
            a = y * Lx + x
            if x < Lx - 1:
                h_edges.append((a, a + 1))
            if y < Ly - 1:
                v_edges.append((a, a + Lx))

    skeleton: list[list[int]] = []
    skeleton.extend([list(e) for e in h_edges if (e[0] % Lx) % 2 == 0])
    skeleton.extend([list(e) for e in h_edges if (e[0] % Lx) % 2 == 1])
    skeleton.extend([list(e) for e in v_edges if (e[0] // Lx) % 2 == 0])
    skeleton.extend([list(e) for e in v_edges if (e[0] // Lx) % 2 == 1])
    return skeleton


def two_local_circuit(skeleton: list[list[int]], seed: int | None = None) -> Circuit:
    """Circuit of Haar-random 4×4 unitaries on each pair in skeleton.

    Each gate gets an independent seed drawn from the top-level RNG,
    so global reproducibility is preserved without changing random_unitary's signature.
    """
    rng = np.random.default_rng(seed)
    gates = [
        MatrixGate(
            f"U{i}",
            indices,
            random_unitary(len(indices), seed=int(rng.integers(2**31))),
        )
        for i, indices in enumerate(skeleton)
    ]
    n = max(idx for pair in skeleton for idx in pair) + 1
    return Circuit(gates, num_sites=n)


def nn_pairs(n: int, periodic: bool = False) -> list[list[int]]:
    """Nearest-neighbor index pairs for a 1D chain of n sites.

    Returns [[0,1],[1,2],...,[n-2,n-1]], plus [n-1,0] if periodic=True.
    TODO: generalize to 2D and other geometries.
    """
    pairs = [[i, i + 1] for i in range(n - 1)]
    if periodic and n > 1:
        pairs.append([n - 1, 0])
    return pairs


def ghz_circuit(n: int) -> Circuit:
    """H(0) followed by CNOT(i, i+1) for i in 0..n-2."""
    gates: list[Gate] = [H(0)] + [CNOT([i, i + 1]) for i in range(n - 1)]
    return Circuit(gates, num_sites=n)


def bell_basis_circuit(a: int, b: int, num_sites: int) -> Circuit:
    """CNOT(a→b) then H(a); pre-rotation before Bell-basis measurement of qubits a, b."""
    return Circuit([CNOT([a, b]), H(a)], num_sites=num_sites)


def ghz_cluster_prep_circuit(cluster_sites: list[int], num_sites: int) -> Circuit:
    """Returns a circuit which prepars a GHZ on cluster_sites and zeros elsewhere"""
    root = cluster_sites[0]
    gates: list[Gate] = [H(root)] + [CNOT([root, s]) for s in cluster_sites[1:]]
    return Circuit(gates, num_sites=num_sites)


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
