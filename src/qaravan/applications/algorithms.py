"""Numerical algorithms applied to a Hamiltonian, kept off the Hamiltonian class itself."""

from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
from qaravan.backends.statevector import Statevector
from qaravan.core.circuit import Circuit
from qaravan.core.fermions import FermionicEncoding
from qaravan.core.gates import MatrixGate
from qaravan.core.hamiltonians import (
    FermionicHamiltonian,
    Hamiltonian,
    HamiltonianTerm,
)

# ---------------------------------------------------------------------------
# Eigensolvers
# ---------------------------------------------------------------------------


METHODS = ("naive", "naive_lanczos", "lanczos")


def ground_state(
    hamiltonian: Hamiltonian,
    method: str = "naive",
    fixed_particle_number: int | None = None,
    sector: np.ndarray | None = None,
) -> Statevector:
    """Ground state of a qubit Hamiltonian, including a JW-encoded one. See METHODS.

    fixed_particle_number restricts to the popcount-k sector before diagonalizing --
    valid whenever total occupation is conserved (Heisenberg/XXZ total S^z, any
    number-conserving encoded fermionic model). sector is the general form: explicit
    basis indices, as the fermionic wrappers pass for spin-resolved sectors.
    """
    if fixed_particle_number is not None:
        if sector is not None:
            raise ValueError("Pass at most one of fixed_particle_number, sector.")
        sector = _number_sector(hamiltonian.num_sites, fixed_particle_number)

    if method == "lanczos":
        return _matrix_free_lanczos(hamiltonian, 1, sector, True)
    matrix = (
        hamiltonian.to_matrix()
        if sector is None
        else hamiltonian.to_sector_matrix(sector)
    )
    _, evecs = _lowest_eigenpairs(matrix, method, 1, True)

    ground_vec = evecs[:, 0]
    if sector is None:
        return Statevector(array=ground_vec)
    full_vec = np.zeros(2**hamiltonian.num_sites, dtype=complex)
    full_vec[sector] = ground_vec
    return Statevector(array=full_vec)


def spectrum(
    hamiltonian: Hamiltonian,
    method: str = "naive",
    num_eigenvalues: int | None = None,
    fixed_particle_number: int | None = None,
    sector: np.ndarray | None = None,
) -> np.ndarray:
    """Ascending eigenvalues of a qubit Hamiltonian; num_eigenvalues=None gives all of them.
    Sector arguments as in ground_state."""
    if fixed_particle_number is not None:
        if sector is not None:
            raise ValueError("Pass at most one of fixed_particle_number, sector.")
        sector = _number_sector(hamiltonian.num_sites, fixed_particle_number)

    if method == "lanczos":
        return _matrix_free_lanczos(hamiltonian, num_eigenvalues, sector, False)
    matrix = (
        hamiltonian.to_matrix()
        if sector is None
        else hamiltonian.to_sector_matrix(sector)
    )
    evals, _ = _lowest_eigenpairs(matrix, method, num_eigenvalues, False)
    return evals


def fermionic_ground_state(
    hamiltonian: FermionicHamiltonian,
    encoding: FermionicEncoding,
    method: str = "naive",
    fixed_particle_number: int | None = None,
    fixed_spin_numbers: tuple[int, int] | None = None,
) -> Statevector:
    """Ground state of hamiltonian.encode(encoding), optionally in a particle-number sector.

    fixed_spin_numbers=(n_up, n_down) restricts each spin species separately, using
    encoding.spin_qubits() to locate the two registers -- only meaningful when N_up and
    N_down are separately conserved. Passing both sector arguments requires them to agree.
    """
    encoded = hamiltonian.encode(encoding)
    sector = _fermionic_sector(
        encoding, encoded.num_sites, fixed_particle_number, fixed_spin_numbers
    )
    return ground_state(encoded, method, sector=sector)


def fermionic_spectrum(
    hamiltonian: FermionicHamiltonian,
    encoding: FermionicEncoding,
    method: str = "naive",
    num_eigenvalues: int | None = None,
    fixed_particle_number: int | None = None,
    fixed_spin_numbers: tuple[int, int] | None = None,
) -> np.ndarray:
    """Ascending eigenvalues of hamiltonian.encode(encoding); sector arguments as in
    fermionic_ground_state."""
    encoded = hamiltonian.encode(encoding)
    sector = _fermionic_sector(
        encoding, encoded.num_sites, fixed_particle_number, fixed_spin_numbers
    )
    return spectrum(encoded, method, num_eigenvalues, sector=sector)


# ---------------------------------------------------------------------------
# Sectors
# ---------------------------------------------------------------------------


def _number_sector(num_sites: int, num_particles: int) -> np.ndarray:
    """Computational-basis indices whose popcount is num_particles."""
    indices = np.arange(2**num_sites, dtype=np.int64)
    return np.flatnonzero(np.bitwise_count(indices) == num_particles)


def _fermionic_sector(
    encoding: FermionicEncoding,
    num_sites: int,
    fixed_particle_number: int | None,
    fixed_spin_numbers: tuple[int, int] | None,
) -> np.ndarray | None:
    """Basis indices for the requested sector, or None if unrestricted."""
    if fixed_spin_numbers is None:
        if fixed_particle_number is None:
            return None
        return _number_sector(num_sites, fixed_particle_number)

    num_up, num_down = fixed_spin_numbers
    if fixed_particle_number is not None and num_up + num_down != fixed_particle_number:
        raise ValueError(
            f"fixed_spin_numbers {fixed_spin_numbers} sums to {num_up + num_down}, "
            f"but fixed_particle_number is {fixed_particle_number}."
        )

    up_qubits, down_qubits = encoding.spin_qubits()
    # qubit q is bit num_sites-1-q of the basis index: qubit 0 is the most significant
    up_mask = sum(1 << (num_sites - 1 - qubit) for qubit in up_qubits)
    down_mask = sum(1 << (num_sites - 1 - qubit) for qubit in down_qubits)
    indices = np.arange(2**num_sites, dtype=np.int64)
    in_sector = (np.bitwise_count(indices & up_mask) == num_up) & (
        np.bitwise_count(indices & down_mask) == num_down
    )
    return np.flatnonzero(in_sector)


# ---------------------------------------------------------------------------
# Diagonalization
# ---------------------------------------------------------------------------


def _lowest_eigenpairs(
    matrix: np.ndarray,
    method: str,
    num_eigenvalues: int | None,
    return_eigenvectors: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Ascending eigenvalues and, when asked, the matching eigenvectors as columns.

    num_eigenvalues=None means the whole spectrum and so requires method="naive";
    "naive_lanczos" only ever computes the k extremal pairs.
    """
    if method == "naive":
        if not return_eigenvectors:
            evals = np.linalg.eigvalsh(matrix)
            return evals[:num_eigenvalues], None
        evals, evecs = np.linalg.eigh(matrix)
        return evals[:num_eigenvalues], evecs[:, :num_eigenvalues]

    if method != "naive_lanczos":
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}.")

    if num_eigenvalues is None:
        raise ValueError(
            "method='naive_lanczos' needs num_eigenvalues; use 'naive' for the full spectrum."
        )
    dim = matrix.shape[0]
    if dim <= num_eigenvalues + 1:
        raise ValueError(
            f"method='naive_lanczos' needs a matrix bigger than {num_eigenvalues + 1}; "
            f"got {dim}x{dim}. Use method='naive'."
        )
    ncv = min(dim - 1, max(2 * num_eigenvalues + 1, 20))
    out = eigsh(
        matrix,
        k=num_eigenvalues,
        which="SA",
        ncv=ncv,
        return_eigenvectors=return_eigenvectors,
    )
    if not return_eigenvectors:
        return np.sort(out), None
    evals, evecs = out
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def _matrix_free_lanczos(
    hamiltonian: Hamiltonian,
    num_eigenvalues: int | None,
    sector: np.ndarray | None,
    return_eigenvectors: bool,
):
    """Lanczos over an implicit H: apply hamiltonian.as_observable() term by term to each
    Krylov vector via op_action, never forming the 2^n x 2^n matrix. O(num_terms * 2^n)
    time and O(2^n) memory per matvec, against O(4^n) for both under naive_lanczos.

    Missing pieces: apply_pauli_sum(pauli_sum, statevector) built on statevector.op_action
    (_expectation_pauli_string is the read-only half of that pattern), and a sector-indexed
    Krylov basis so the restriction costs nothing.
    """
    raise NotImplementedError(
        "method='lanczos' (matrix-free) is not implemented yet; use 'naive_lanczos'."
    )


# ---------------------------------------------------------------------------
# Trotter
# ---------------------------------------------------------------------------


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


def _trotter_gate(term: HamiltonianTerm, dt: float) -> MatrixGate:
    """exp(-i dt O) for the term's local observable O, named after O itself.

    The name is what shows up in reprs and circuit diagrams, so it carries the generator
    (e.g. '-1 ZZ', 'XX + YY + ZZ') rather than a generic label. Following RZ/RZZ/XY, the gate
    is named for the operator it exponentiates, not the exponential.
    """
    mat = expm(-1j * dt * term.local_obs.to_matrix())
    return MatrixGate(repr(term.local_obs), term.sites, mat)


def _trotter_order1(groups: list[list[HamiltonianTerm]], dt: float) -> list[MatrixGate]:
    """Apply each group once at step size dt."""
    gates: list[MatrixGate] = []
    for group in groups:
        for term in group:
            gates.append(_trotter_gate(term, dt))
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
            gates.append(_trotter_gate(term, half_dt))
    # last group: full step
    for term in groups[-1]:
        gates.append(_trotter_gate(term, dt))
    # backward half-steps: reverse of all but last group
    for group in reversed(groups[:-1]):
        for term in group:
            gates.append(_trotter_gate(term, half_dt))
    return gates
