"""Entanglement measures, circuit complexity bounds and distances for statevectors and unitaries."""

from __future__ import annotations
import numpy as np
from qaravan.backends.statevector import Statevector
from qaravan.core.circuit import Circuit

Arraylike = Statevector | Circuit | np.ndarray


def entanglement_entropy(sv: Statevector, sites: list[int]) -> float:
    """Von Neumann entropy S = -Tr[ρ_A log2 ρ_A] of the reduced state on sites, in bits.

    Eigenvalues below 1e-12 are treated as numerical zeros and excluded from the sum.
    """
    rdm = sv.rdm(sites)
    eigenvalues = np.linalg.eigvalsh(rdm)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    eigenvalues /= eigenvalues.sum()
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


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

    For a depth-d 1D brickwall each cut accumulates at most d bits. Returns ceil(max_k S_k).
    """
    profile = entanglement_profile(sv)
    return int(np.ceil(profile.max()))


def phase_free_distance(a: Arraylike, b: Arraylike) -> float:
    """||a - e^{i phi} b|| with phi aligning the overlap <b|a>, i.e. the distance up to a global phase.

    Statevectors and 1D arrays use the 2-norm, where this phi is the exact minimiser and the result
    is sqrt(2 - 2|<b|a>|) for unit vectors. Circuits (via to_matrix) and 2D arrays use the spectral
    norm, with phi aligned on the Hilbert-Schmidt overlap Tr(b^† a): an upper bound on the minimum
    over phi. Raises TypeError if a and b are not the same type, ValueError if their shapes differ.
    """
    if type(a) is not type(b):
        raise TypeError(f"a and b must be the same type, got {type(a).__name__} and {type(b).__name__}")
    u, v = _as_array(a), _as_array(b)
    if u.shape != v.shape:
        raise ValueError(f"shapes differ: {u.shape} vs {v.shape}")
    phase = np.exp(1j * np.angle(np.vdot(v, u)))
    return float(np.linalg.norm(u - phase * v, ord=2))


def _as_array(x: Arraylike) -> np.ndarray:
    if isinstance(x, Statevector):
        return x.to_array()
    if isinstance(x, Circuit):
        return x.to_matrix()
    return np.asarray(x)
