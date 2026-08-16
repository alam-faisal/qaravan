"""Numerical algorithms applied to a Hamiltonian, kept off the Hamiltonian class itself.

Hamiltonian stays a pure data container (terms, trotter_groups, and the intrinsic
representations as_matrix()/as_observable()) plus __repr__ -- nothing that "runs an
algorithm" is a method on it. Two different reasons land a function here rather than in
core/hamiltonians.py:

- ground_state() *has* to be here: it hands back a Statevector, and core/ must not depend on
  backends/ (backends/ already depends on core/) -- putting it on Hamiltonian caused a real
  circular import that only bit depending on which module a caller happened to import first.
- trotter_circuit() doesn't have that problem -- it returns a plain core/ Circuit, no backend
  involved. It moved here anyway, for symmetry with ground_state: Suzuki-Trotter
  decomposition is "an algorithm applied to a Hamiltonian," the same shape as diagonalization,
  and Hamiltonian carrying no algorithm methods at all is a simpler, more consistent line to
  hold than carrying some and not others.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh

from qaravan.backends.statevector import Statevector
from qaravan.core.circuit import Circuit
from qaravan.core.gates import MatrixGate
from qaravan.core.hamiltonians import Hamiltonian, HamiltonianTerm


def ground_state(
    hamiltonian: Hamiltonian,
    method: str = "exact",
    fixed_particle_number: int | None = None,
) -> Statevector:
    """Ground state of hamiltonian's full 2^n x 2^n matrix.

    method="exact" (default): full dense diagonalization (numpy.linalg.eigh). Computes the
    whole spectrum even though only the bottom eigenvector is kept -- O(n^3), but robust at
    any size, including the tiny systems (n<20) most tests use.
    method="lanczos": scipy.sparse.linalg.eigsh(k=1) -- only the extremal eigenpair,
    O(n^2) per iteration instead of diagonalizing everything. Much faster once n is large
    enough that "compute everything" is wasteful; raises ValueError if the (possibly
    sector-restricted) matrix is too small for that to make sense -- use "exact" there.
    For a degenerate ground level, either method returns *some* vector spanning the ground
    subspace, not necessarily the same one run to run.

    fixed_particle_number restricts to the popcount-k sector of the computational basis
    before diagonalizing -- meaningful whenever the Hamiltonian conserves total
    occupation number, e.g. any Jordan-Wigner-encoded fermionic Hamiltonian.
    """
    H = hamiltonian.as_matrix()
    num_sites = hamiltonian.num_sites
    if fixed_particle_number is None:
        sector = None
        H_reduced = H
    else:
        sector = [
            i for i in range(2**num_sites) if bin(i).count("1") == fixed_particle_number
        ]
        H_reduced = H[np.ix_(sector, sector)]

    if method == "exact":
        _, evecs = np.linalg.eigh(H_reduced)
        ground_vec = evecs[:, 0]
    elif method == "lanczos":
        n = H_reduced.shape[0]
        ncv = min(n - 1, max(3, 20))
        if ncv <= 1:
            raise ValueError(
                f"method='lanczos' needs a larger matrix than {n}x{n}; use method='exact'."
            )
        _, evecs = eigsh(H_reduced, k=1, which="SA", ncv=ncv)
        ground_vec = evecs[:, 0]
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'exact' or 'lanczos'.")

    if sector is None:
        return Statevector(array=ground_vec)
    full_vec = np.zeros(2**num_sites, dtype=complex)
    for local_idx, global_idx in enumerate(sector):
        full_vec[global_idx] = ground_vec[local_idx]
    return Statevector(array=full_vec)


def trotter_circuit(hamiltonian: Hamiltonian, dt: float, order: int = 2) -> Circuit:
    """Single Trotter step of duration dt for hamiltonian.

    order=1: sequential groups at full dt.
    order=2: symmetric (Suzuki-Trotter) with boundary half-steps.
    Raises NotImplementedError if hamiltonian.trotter_groups is None.
    Raises ValueError for systems too small to have distinct even/odd bond layers.
    """
    if hamiltonian.trotter_groups is None:
        raise NotImplementedError(
            f"{type(hamiltonian).__name__} has no trotter_groups defined."
        )
    if not hamiltonian.gate_fusion_applied:
        warnings.warn(
            "This Hamiltonian's terms were produced by a generic fermion-to-qubit "
            "encoding with no gate-fusion optimization pass -- the Trotter circuit may "
            "use more gates than necessary (e.g. commuting diagonal terms on the same "
            "qubits are not merged into one gate). Correctness and Trotter error "
            "scaling are unaffected.",
            stacklevel=2,
        )
    if hamiltonian.num_sites <= 2:
        raise ValueError(
            f"trotter_circuit requires num_sites > 2; got {hamiltonian.num_sites}."
        )
    groups = [g for g in hamiltonian.trotter_groups if g]  # skip empty groups

    if order == 1:
        gates = _trotter_order1(groups, dt)
    elif order == 2:
        gates = _trotter_order2(groups, dt)
    else:
        raise ValueError(f"Trotter order must be 1 or 2; got {order}.")

    return Circuit(gates, num_sites=hamiltonian.num_sites)


def _trotter_order1(groups: list[list[HamiltonianTerm]], dt: float) -> list[MatrixGate]:
    """Apply each group once at step size dt."""
    gates: list[MatrixGate] = []
    for group in groups:
        for term in group:
            mat = expm(-1j * dt * term.local_obs.matrix)
            gates.append(MatrixGate("Trotter", term.sites, mat))
    return gates


def _trotter_order2(groups: list[list[HamiltonianTerm]], dt: float) -> list[MatrixGate]:
    """Symmetric second-order Trotter for k groups.

    Pattern: g0(dt/2) g1(dt/2) ... g_{k-2}(dt/2) g_{k-1}(dt) g_{k-2}(dt/2) ... g0(dt/2)
    """
    gates: list[MatrixGate] = []
    half_dt = dt / 2.0
    # forward half-steps: all but last group
    for group in groups[:-1]:
        for term in group:
            mat = expm(-1j * half_dt * term.local_obs.matrix)
            gates.append(MatrixGate("Trotter", term.sites, mat))
    # last group: full step
    for term in groups[-1]:
        mat = expm(-1j * dt * term.local_obs.matrix)
        gates.append(MatrixGate("Trotter", term.sites, mat))
    # backward half-steps: reverse of all but last group
    for group in reversed(groups[:-1]):
        for term in group:
            mat = expm(-1j * half_dt * term.local_obs.matrix)
            gates.append(MatrixGate("Trotter", term.sites, mat))
    return gates
