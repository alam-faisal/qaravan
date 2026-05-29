"""Entanglement measures and circuit complexity bounds for statevectors."""

from __future__ import annotations

import numpy as np

from qaravan.backends.statevector import Statevector


def entanglement_entropy(sv: Statevector, sites: list[int]) -> float:
    """Von Neumann entropy S = -Tr[ρ_A log ρ_A] of the reduced state on sites.

    Eigenvalues below 1e-12 are treated as numerical zeros and excluded from the sum.
    """
    rdm = sv.rdm(sites)
    eigenvalues = np.linalg.eigvalsh(rdm)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    eigenvalues /= eigenvalues.sum()
    return float(-np.sum(eigenvalues * np.log(eigenvalues)))


def entanglement_profile(sv: Statevector) -> np.ndarray:
    """Bipartite entropy S([0..k] | [k+1..n-1]) for each contiguous cut k=0..n-2.

    Returns array of length n-1.
    """
    n = sv.num_sites
    return np.array(
        [entanglement_entropy(sv, list(range(k + 1))) for k in range(n - 1)]
    )


def entanglement_depth(sv: Statevector) -> int:
    """Lower bound on nearest-neighbour brickwall depth to generate sv's entanglement.

    For a depth-d 1D brickwall each cut accumulates at most d * log(2) entropy.
    Returns ceil(max_k S_k / log(2)).
    """
    profile = entanglement_profile(sv)
    return int(np.ceil(profile.max() / np.log(2)))
