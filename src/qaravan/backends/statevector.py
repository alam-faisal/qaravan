"""Statevector backend — Statevector(State) and StatevectorSimulator(Simulator)."""

from __future__ import annotations

import copy

import numpy as np
from ncon_torch import ncon, permute

from qaravan.core.base import (
    Gate,
    IncompatibleStateError,
    Simulator,
    State,
)
from qaravan.core.observables import PAULI_MATRICES, LocalOp, PauliString, PauliSum


# ---------------------------------------------------------------------------
# ncon primitives (ported from legacy/statevector_sim.py, no semantic changes)
# ---------------------------------------------------------------------------


def _locs_to_indices(locs: list[int], n: int) -> tuple[list[int], list[int]]:
    """ncon index labels for a gate at locs on a rank-n state tensor."""
    shifted_locs = [loc + 1 for loc in locs]
    gate_indices = [-i for i in shifted_locs] + shifted_locs

    boundaries = [0] + shifted_locs
    tensor_indices: list[int] = []
    for i in range(len(shifted_locs)):
        tensor_indices += [-j for j in range(boundaries[i] + 1, boundaries[i + 1])]
        tensor_indices.append(shifted_locs[i])
    tensor_indices += [-j for j in range(boundaries[-1] + 1, n + 1)]
    return gate_indices, tensor_indices


def op_action(
    op: np.ndarray,
    indices: list[int],
    sv: np.ndarray,
    local_dim: int = 2,
) -> np.ndarray:
    """Apply op at indices to rank-n tensor sv; handles non-contiguous / out-of-order sites.

    op: d^k × d^k matrix or rank-2k tensor for a k-site gate.
    Returns same shape as sv.
    """
    if op.ndim != 2 * len(indices):
        op = op.reshape(*[local_dim] * 2 * len(indices))

    n = sv.ndim if sv.ndim > 1 else int(np.log(len(sv)) / np.log(local_dim))
    state = sv.reshape(*[local_dim] * n) if sv.ndim == 1 else sv

    sorted_indices = sorted(indices)
    sort_order = [indices.index(i) for i in sorted_indices]
    perm = sort_order + [i + len(indices) for i in sort_order]
    op = permute(op, perm)

    gate_indices, state_indices = _locs_to_indices(sorted_indices, n)
    new_sv = ncon((op, state), (gate_indices, state_indices))
    return new_sv.reshape(local_dim**n) if sv.ndim == 1 else new_sv


def partial_overlap(
    sv1: np.ndarray,
    sv2: np.ndarray,
    local_dim: int = 2,
    skip: list[int] | None = None,
) -> np.ndarray:
    """Partial ⟨sv1|sv2⟩ contracting all sites not in skip.

    sv1, sv2: flat statevectors, shape (local_dim**n,).
    skip=None or skip=[] → full overlap, returns (1, 1) array.
    Returns (local_dim**|skip|, local_dim**|skip|) matrix.
    Specifically: result[i,j] = ⟨sv2[j-block]|sv1[i-block]⟩ in the skip subspace.
    """
    system_size = int(np.log(len(sv1)) / np.log(local_dim))
    sites = sorted(skip) if skip is not None else []

    psi = sv1.reshape([local_dim] * system_size)
    phi_conj = sv2.reshape([local_dim] * system_size).conj()

    psi_labels = [0] * system_size
    phi_conj_labels = [0] * system_size

    next_contract_label = 1
    next_free_label = -1

    for i in range(system_size):
        if i in sites:
            psi_labels[i] = next_free_label
            phi_conj_labels[i] = next_free_label - len(sites)
            next_free_label -= 1
        else:
            psi_labels[i] = next_contract_label
            phi_conj_labels[i] = next_contract_label
            next_contract_label += 1

    contraction = ncon([psi, phi_conj], [psi_labels, phi_conj_labels])
    kept_dim = local_dim ** len(sites)
    return contraction.reshape((kept_dim, kept_dim))


# ---------------------------------------------------------------------------
# Repr helper
# ---------------------------------------------------------------------------


def _fmt_coeff(a: complex, threshold: float = 1e-9) -> str:
    """Format a complex amplitude for display: real-only, imag-only, or (r+ij)."""
    r, i = a.real, a.imag
    if abs(i) < threshold:
        return f"{r:.4f}"
    if abs(r) < threshold:
        return f"{i:.4f}i"
    return f"({r:.4f}{i:+.4f}i)"


# ---------------------------------------------------------------------------
# Expectation dispatch helpers
# ---------------------------------------------------------------------------


def _expectation_pauli_string(
    tensor: np.ndarray,
    obs: PauliString,
    local_dim: int = 2,
) -> complex:
    """⟨ψ|P₀⊗P₁⊗…|ψ⟩ using per-site op_action; never builds the 2^n matrix."""
    n = obs.num_sites
    flat = tensor.reshape(local_dim**n)
    right_tensor = copy.deepcopy(tensor)
    for i, p in enumerate(obs.string):
        if p == "I":
            continue
        right_tensor = op_action(PAULI_MATRICES[p], [i], right_tensor, local_dim)
    result = partial_overlap(right_tensor.reshape(local_dim**n), flat)[0, 0]
    return obs.coeff * result


def _expectation_local_op(
    tensor: np.ndarray,
    obs: LocalOp,
    local_dim: int = 2,
) -> complex:
    """⟨ψ|O_sites|ψ⟩ for a LocalOp on obs.indices."""
    n = tensor.ndim
    flat = tensor.reshape(local_dim**n)
    right_tensor = op_action(
        obs.matrix, list(obs.indices), copy.deepcopy(tensor), local_dim
    )
    return partial_overlap(right_tensor.reshape(local_dim**n), flat)[0, 0]


# ---------------------------------------------------------------------------
# Statevector
# ---------------------------------------------------------------------------


class Statevector(State):
    """Pure statevector backend. Internal state: rank-n complex128 tensor."""

    def __init__(
        self,
        num_sites: int | None = None,
        *,
        bitstring: str | None = None,
        array: np.ndarray | None = None,
        random_seed: int | None = None,
        local_dim: int = 2,
    ):
        paths = [num_sites is not None, bitstring is not None, array is not None]
        if sum(paths) != 1:
            raise ValueError(
                "Exactly one of num_sites, bitstring, or array must be provided."
            )

        self.local_dim = local_dim

        if bitstring is not None:
            self.num_sites = len(bitstring)
            idx = int(bitstring, local_dim)
            t = np.zeros([local_dim] * self.num_sites, dtype=complex)
            t.flat[idx] = 1.0
            self._tensor = t

        elif array is not None:
            arr = np.asarray(array, dtype=complex).reshape(-1)
            if not np.isclose(np.linalg.norm(arr), 1.0, atol=1e-6):
                raise ValueError("Provided array must be normalized.")
            n = round(np.log(arr.size) / np.log(local_dim))
            self.num_sites = n
            self._tensor = arr.reshape([local_dim] * n)

        else:  # num_sites path
            self.num_sites = num_sites  # type: ignore[assignment]
            if random_seed is not None:
                rng = np.random.default_rng(random_seed)
                arr = rng.standard_normal(
                    local_dim**num_sites
                ) + 1j * rng.standard_normal(local_dim**num_sites)
                arr = arr / np.linalg.norm(arr)
                self._tensor = arr.reshape([local_dim] * num_sites)
            else:
                arr = np.zeros(local_dim**num_sites, dtype=complex)
                arr[0] = 1.0
                self._tensor = arr.reshape([local_dim] * num_sites)

    # ------------------------------------------------------------------ ABC

    @property
    def default_simulator(self) -> type[Simulator]:
        return StatevectorSimulator

    def expectation(self, observable) -> complex:
        """Dispatch on observable type; never builds the full 2^n matrix."""
        if isinstance(observable, PauliString):
            if observable.num_sites != self.num_sites:
                raise ValueError(
                    f"PauliString length {observable.num_sites} != num_sites {self.num_sites}"
                )
            return _expectation_pauli_string(self._tensor, observable, self.local_dim)
        if isinstance(observable, PauliSum):
            if observable.num_sites != self.num_sites:
                raise ValueError(
                    f"PauliSum length {observable.num_sites} != num_sites {self.num_sites}"
                )
            return sum(
                _expectation_pauli_string(self._tensor, t, self.local_dim)
                for t in observable.terms
            )
        if isinstance(observable, LocalOp):
            return _expectation_local_op(self._tensor, observable, self.local_dim)
        raise NotImplementedError(
            f"Statevector.expectation: unsupported observable type {type(observable)}"
        )

    def sample(self, num_shots: int) -> np.ndarray:
        """Born-rule sampling of full bitstrings; returns (num_shots, num_sites) int8 array."""
        probs = np.abs(self.to_array()) ** 2
        flat = np.random.default_rng().choice(len(probs), size=num_shots, p=probs)
        bits = (flat[:, None] >> np.arange(self.num_sites - 1, -1, -1)) & 1
        return bits.astype(np.int8)

    def measure_and_collapse(self, sites: list[int]) -> tuple[Statevector, str]:
        """Sample outcome from rdm diagonal, project+renorm; returned state has same num_sites."""
        sorted_sites = sorted(sites)
        probs = np.real(np.diag(self.rdm(sorted_sites)))
        outcome_idx = np.random.default_rng().choice(len(probs), p=probs)
        outcome_str = format(outcome_idx, f"0{len(sites)}b")
        return self.project_and_renorm(sorted_sites, outcome_str), outcome_str

    def partial_overlap(self, other: Statevector, skip: list[int]) -> np.ndarray:
        """result[i,j] = ⟨other[j]|self[i]⟩ in the skip subspace.

        Wraps module-level partial_overlap (resolved via global scope, not class namespace).
        For self==other: equals rdm(skip). Full overlap: [0,0].conj() == overlap(other).
        """
        return partial_overlap(self.to_array(), other.to_array(), local_dim=self.local_dim, skip=skip)

    # ------------------------------------------------------------------ Extra public

    def rdm(self, sites: list[int]) -> np.ndarray:
        """Reduced density matrix for sites; (local_dim**|sites|, local_dim**|sites|)."""
        flat = self.to_array()
        return partial_overlap(flat, flat, local_dim=self.local_dim, skip=sites)

    def project_and_renorm(self, sites: list[int], outcome_str: str) -> Statevector:
        """Apply |bit_i⟩⟨bit_i| projector at each site via op_action, then renorm."""
        tensor = copy.deepcopy(self._tensor)
        for site, bit in zip(sites, outcome_str):
            b = int(bit)
            proj = np.zeros((self.local_dim, self.local_dim), dtype=complex)
            proj[b, b] = 1.0
            tensor = op_action(proj, [site], tensor, self.local_dim)
        flat = tensor.reshape(self.local_dim**self.num_sites)
        norm = np.linalg.norm(flat)
        if norm < 1e-12:
            raise ValueError(
                f"Projection onto '{outcome_str}' at sites {sites} gives zero state."
            )
        return Statevector(array=flat / norm, local_dim=self.local_dim)

    def reset(self, sites: list[int], reset_to: int | list[int] = 0) -> Statevector:
        """Measure sites, then conditionally flip to reset_to; returned state has same num_sites."""
        if isinstance(reset_to, int):
            targets = [reset_to] * len(sites)
        else:
            targets = list(reset_to)

        # sort sites and targets together so measure_and_collapse outcome aligns
        sorted_pairs = sorted(zip(sites, targets))
        sorted_sites = [s for s, _ in sorted_pairs]
        sorted_targets = [t for _, t in sorted_pairs]

        sv_post, outcome_str = self.measure_and_collapse(sorted_sites)

        tensor = sv_post._tensor.copy()
        for site, bit, target in zip(sorted_sites, outcome_str, sorted_targets):
            if int(bit) != target:
                x_mat = np.array([[0, 1], [1, 0]], dtype=complex)
                tensor = op_action(x_mat, [site], tensor, self.local_dim)

        return Statevector(array=tensor.reshape(-1), local_dim=self.local_dim)

    def norm(self) -> float:
        """Should be 1.0 for a valid state."""
        flat = self.to_array()
        return partial_overlap(flat, flat, self.local_dim)[0, 0].real

    def to_array(self) -> np.ndarray:
        """Flat statevector, shape (local_dim**num_sites,), C-order."""
        return self._tensor.reshape(self.local_dim**self.num_sites)

    def __repr__(self) -> str:
        d, n = self.local_dim, self.num_sites
        terms = []
        for idx, amp in enumerate(self.to_array()):
            if abs(amp) < 1e-9:
                continue
            digits, tmp = [], idx
            for _ in range(n):
                digits.append(tmp % d)
                tmp //= d
            ket = "".join(map(str, reversed(digits)))
            terms.append(f"{_fmt_coeff(amp)}|{ket}⟩")
        return " + ".join(terms) if terms else "0"


# ---------------------------------------------------------------------------
# StatevectorSimulator
# ---------------------------------------------------------------------------


class StatevectorSimulator(Simulator):
    """Evolves a Statevector under a Circuit using op_action."""

    def _validate_state(self, state: State) -> None:
        if not isinstance(state, Statevector):
            raise IncompatibleStateError(
                f"StatevectorSimulator requires a Statevector; got {type(state).__name__}"
            )

    def translate_gate(self, gate: Gate) -> tuple[np.ndarray, list[int]]:
        return gate.matrix, gate.indices

    def _apply_translated_gate(
        self, state: Statevector, translated_gate: tuple[np.ndarray, list[int]]
    ) -> None:
        mat, indices = translated_gate
        state._tensor = op_action(mat, indices, state._tensor, state.local_dim)
