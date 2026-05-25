"""Common circuit instances and parametric layer factories."""

from __future__ import annotations

import numpy as np

from qaravan.core.circuits import Circuit
from qaravan.core.gates import CNOT, H, Gate, RX, RY, RZ, RXX, RYY, RZZ


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
