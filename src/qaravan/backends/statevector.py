"""Statevector backend — Statevector(State) and StatevectorSimulator(Simulator)."""

from __future__ import annotations
import copy
from typing import TYPE_CHECKING
import numpy as np
from qaravan.backends._ncon_compat import ncon, permute
from qaravan.core._repr import (
    as_math,
    elide,
    format_coefficient,
    join_terms,
    latex_coefficient,
    math_more,
)
from qaravan.core.base import (
    Gate,
    IncompatibleNoiseError,
    IncompatibleStateError,
    NoiseModel,
    Simulator,
    State,
)
from qaravan.core.gates import pauli_string_to_gates
from qaravan.core.observable import LocalOp
from qaravan.core.pauli import PAULI_MATRICES, PauliString, PauliSum

if TYPE_CHECKING:
    from qaravan.core.circuit import Circuit

# === Tensor contraction helpers ===


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


# === Expectation helpers ===


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
        obs.to_matrix(), list(obs.indices), copy.deepcopy(tensor), local_dim
    )
    return partial_overlap(right_tensor.reshape(local_dim**n), flat)[0, 0]


def apply_pauli_sum(
    pauli_sum: PauliSum,
    vector: np.ndarray,
    sector: np.ndarray | None = None,
) -> np.ndarray:
    """H|v⟩ via each term's bitmasks, never forming the 2^n × 2^n matrix.

    vector and the result are flat arrays over the same index set: computational-basis
    indices, or positions within sector (ascending basis indices) when one is given. A sector
    confines the whole operation to that subspace, so cost and memory scale with len(sector).
    Images leaving it are dropped, exact only for an operator that preserves it -- see
    PauliSum.to_sector_matrix.
    """
    basis = (
        np.arange(2**pauli_sum.num_sites, dtype=np.int64) if sector is None else sector
    )
    out = np.zeros(len(basis), dtype=complex)
    for term in pauli_sum.terms:
        x_mask, z_mask, n_y = term.bitmasks()
        phase = term.coeff * (1j**n_y) * (-1.0) ** np.bitwise_count(basis & z_mask)
        if sector is None:
            out[basis ^ x_mask] += phase * vector
            continue
        images = basis ^ x_mask
        rows = np.searchsorted(basis, images).clip(0, len(basis) - 1)
        inside = basis[rows] == images
        out[rows[inside]] += phase[inside] * vector[inside]
    return out


# === Statevector ===


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

    def sample(
        self, num_shots: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Born-rule sampling of full bitstrings; returns (num_shots, num_sites) int8 array."""
        probs = np.abs(self.to_array()) ** 2
        flat = np.random.default_rng(rng).choice(len(probs), size=num_shots, p=probs)
        bits = (flat[:, None] >> np.arange(self.num_sites - 1, -1, -1)) & 1
        return bits.astype(np.int8)

    def measure_and_collapse(
        self, sites: list[int], rng: np.random.Generator | None = None
    ) -> tuple[Statevector, str]:
        """Sample outcome from rdm diagonal, project+renorm; returned state has same num_sites."""
        sorted_sites = sorted(sites)
        probs = np.real(np.diag(self.rdm(sorted_sites)))
        outcome_idx = np.random.default_rng(rng).choice(len(probs), p=probs)
        outcome_str = format(outcome_idx, f"0{len(sites)}b")
        return self.project_and_renorm(sorted_sites, outcome_str), outcome_str

    def partial_overlap(self, other: Statevector, skip: list[int]) -> np.ndarray:
        """result[i,j] = ⟨other[j]|self[i]⟩ in the skip subspace.

        Wraps module-level partial_overlap (resolved via global scope, not class namespace).
        For self==other: equals rdm(skip). Full overlap: [0,0].conj() == overlap(other).
        """
        return partial_overlap(
            self.to_array(), other.to_array(), local_dim=self.local_dim, skip=skip
        )

    def rdm(self, sites: list[int]) -> np.ndarray:
        """Reduced density matrix for sites; (local_dim**|sites|, local_dim**|sites|)."""
        flat = self.to_array()
        return partial_overlap(flat, flat, local_dim=self.local_dim, skip=sites)

    def drop_sites(self, sites: list[int], outcome: str) -> Statevector:
        """Return the pure-state factor on remaining qubits given classical values at sites.

        For a product state |ψ⟩_{remaining} ⊗ |outcome⟩_{sites}, returns |ψ⟩_{remaining}
        as a Statevector with num_sites = self.num_sites - len(sites).
        sites must be sorted to match the digit order of outcome.
        """
        idx: list[int | slice] = [slice(None)] * self.num_sites
        for site, bit in zip(sorted(sites), outcome):
            idx[site] = int(bit)
        sub = self._tensor[tuple(idx)].reshape(-1)
        norm = np.linalg.norm(sub)
        if norm < 1e-12:
            raise ValueError(
                f"Slice at {sites}='{outcome}' gives zero amplitude — sites may not be in that state."
            )
        return Statevector(array=sub / norm, local_dim=self.local_dim)

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
                tensor = op_action(PAULI_MATRICES["X"], [site], tensor, self.local_dim)

        return Statevector(array=tensor.reshape(-1), local_dim=self.local_dim)

    def norm(self) -> float:
        """Should be 1.0 for a valid state."""
        flat = self.to_array()
        return partial_overlap(flat, flat, self.local_dim)[0, 0].real

    def to_array(self) -> np.ndarray:
        """Flat statevector, shape (local_dim**num_sites,), C-order."""
        return self._tensor.reshape(self.local_dim**self.num_sites)

    def _nonzero_amplitudes(self, threshold: float = 1e-9) -> list[tuple[str, complex]]:
        """(ket digits, amplitude) for the populated basis states, most significant digit first."""
        d, n = self.local_dim, self.num_sites
        populated = []
        for index, amp in enumerate(self.to_array()):
            if abs(amp) < threshold:
                continue
            digits, remainder = [], index
            for _ in range(n):
                digits.append(remainder % d)
                remainder //= d
            populated.append(("".join(map(str, reversed(digits))), amp))
        return populated

    def __repr__(self) -> str:
        parts = [
            f"{format_coefficient(amp, spec='.4f')}|{ket}⟩"
            for ket, amp in self._nonzero_amplitudes()
        ]
        return join_terms(elide(parts))

    def _repr_latex_(self) -> str:
        parts = [
            f"{latex_coefficient(amp, '.4f')} \\, \\lvert {ket} \\rangle"
            for ket, amp in self._nonzero_amplitudes()
        ]
        return as_math(join_terms(elide(parts, marker=math_more)))


# === StatevectorSimulator ===


class StatevectorSimulator(Simulator):
    """Evolves a Statevector under a Circuit using op_action."""

    def _validate_state(self, state: State) -> None:
        if not isinstance(state, Statevector):
            raise IncompatibleStateError(
                f"StatevectorSimulator requires a Statevector; got {type(state).__name__}"
            )

    def _validate_noise(self, noise_model: NoiseModel) -> None:
        if not noise_model.implements("sample_error"):
            raise IncompatibleNoiseError(
                f"StatevectorSimulator needs sampled Pauli errors; "
                f"{type(noise_model).__name__} does not implement sample_error"
            )

    def translate_gate(self, gate: Gate) -> tuple[np.ndarray, list[int]]:
        return gate.matrix, gate.indices

    def _apply_translated_gate(
        self, state: Statevector, translated_gate: tuple[np.ndarray, list[int]]
    ) -> None:
        mat, indices = translated_gate
        state._tensor = op_action(mat, indices, state._tensor, state.local_dim)

    def _insert_noise(
        self,
        circuit: Circuit,
        noise_model: NoiseModel | None,
    ) -> Circuit:
        """Return a circuit with freshly sampled Pauli-error layers."""
        if noise_model is None:
            return circuit

        realized = circuit.copy()
        assert circuit.layers is not None
        if noise_model.placement == "circuit":
            realized.layers = list(circuit.layers)
            noise_gates = pauli_string_to_gates(
                noise_model.sample_error(self.noise_rng)
            )
            if noise_gates:
                realized.layers.append(noise_gates)
            return realized
        if noise_model.placement != "layer":
            raise ValueError(
                f"unknown noise placement {noise_model.placement!r}; use 'layer' or 'circuit'"
            )

        realized.layers = []
        for layer in circuit.layers:
            realized.layers.append(layer)
            noise_gates = pauli_string_to_gates(
                noise_model.sample_error(self.noise_rng)
            )
            if noise_gates:
                realized.layers.append(noise_gates)
        return realized
