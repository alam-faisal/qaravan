"""Lattice connectivity helpers for Hamiltonian construction."""

from __future__ import annotations


class LinearLattice:
    """1D chain: provides site connectivity for Hamiltonian construction."""

    def __init__(self, n: int, periodic: bool = False):
        if n < 2:
            raise ValueError(f"LinearLattice requires n >= 2; got {n}.")
        self.n = n
        self.periodic = periodic

    def nn_pairs(self) -> list[tuple[int, int]]:
        """Nearest-neighbor pairs. Open chain: n-1 bonds. Periodic ring: n bonds."""
        pairs = [(i, i + 1) for i in range(self.n - 1)]
        if self.periodic:
            pairs.append((self.n - 1, 0))
        return pairs
