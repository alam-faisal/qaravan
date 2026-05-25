"""Concrete Gate subclasses: MatrixGate, ParametricGate, and all named gates."""

from __future__ import annotations
import numpy as np
from abc import abstractmethod, ABC
from qaravan.core.base import Gate


# ---------------------------------------------------------------------------
# MatrixGate and ParametricGate — base concrete classes
# ---------------------------------------------------------------------------


class MatrixGate(Gate):
    """Concrete Gate with a fixed matrix stored at construction.

    Use for ad-hoc gates: MatrixGate("U", [0, 1], some_4x4).
    """

    def __init__(
        self,
        name: str,
        indices: int | list[int],
        matrix: np.ndarray,
        time: float | None = None,
    ):
        super().__init__(name, indices, time)
        self._matrix = np.asarray(matrix, dtype=complex)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    def dagger(self) -> MatrixGate:
        return MatrixGate(
            self.name + "†", self.indices, self._matrix.conj().T, self.time
        )


class ParametricGate(Gate, ABC):
    """Abstract gate whose matrix is computed from continuous parameters.

    Subclasses implement _build_matrix(). params is a tuple of floats.
    dagger() negates all params — correct for all exp(-i θ P) rotation gates.
    """

    def __init__(
        self,
        name: str,
        indices: int | list[int],
        params: tuple[float, ...],
        time: float | None = None,
    ):
        super().__init__(name, indices, time)
        self.params = params
        self.num_params = len(params)

    @abstractmethod
    def _build_matrix(self) -> np.ndarray: ...

    @property
    def matrix(self) -> np.ndarray:
        return self._build_matrix()

    def dagger(self) -> ParametricGate:
        return type(self)(self.indices, *(-p for p in self.params), time=self.time)

    def __str__(self) -> str:
        params_str = ", ".join(f"{p:.4g}" for p in self.params)
        return f"{self.name}({self.indices}, {params_str})"

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# Single-qubit MatrixGate subclasses
# ---------------------------------------------------------------------------


class I(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__("I", indices, np.eye(2, dtype=complex), time)


class X(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__("X", indices, np.array([[0, 1], [1, 0]], dtype=complex), time)


class Y(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__("Y", indices, np.array([[0, -1j], [1j, 0]]), time)


class Z(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__("Z", indices, np.array([[1, 0], [0, -1]], dtype=complex), time)


class H(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__("H", indices, np.array([[1, 1], [1, -1]]) / np.sqrt(2), time)


class S(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__("S", indices, np.array([[1, 0], [0, 1j]]), time)

    def dagger(self) -> Sdg:
        return Sdg(self.indices, self.time)


class Sdg(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__("Sdg", indices, np.array([[1, 0], [0, -1j]]), time)

    def dagger(self) -> S:
        return S(self.indices, self.time)


class T(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__(
            "T", indices, np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]), time
        )

    def dagger(self) -> Tdg:
        return Tdg(self.indices, self.time)


class Tdg(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__(
            "Tdg", indices, np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]]), time
        )

    def dagger(self) -> T:
        return T(self.indices, self.time)


class SX(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__(
            "SX", indices, (1 / np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]]), time
        )


# ---------------------------------------------------------------------------
# Single-qubit ParametricGate subclasses
# Convention: exp(-i θ P), full angle — no hidden 1/2 factor.
# ---------------------------------------------------------------------------


class RX(ParametricGate):
    """exp(-i θ X) = cos(θ)I - i sin(θ)X."""

    def __init__(
        self, indices: int | list[int], theta: float, time: float | None = None
    ):
        super().__init__("RX", indices, (theta,), time)

    def _build_matrix(self) -> np.ndarray:
        c, s = np.cos(self.params[0]), np.sin(self.params[0])
        return np.array([[c, -1j * s], [-1j * s, c]])


class RY(ParametricGate):
    """exp(-i θ Y) = cos(θ)I - i sin(θ)Y."""

    def __init__(
        self, indices: int | list[int], theta: float, time: float | None = None
    ):
        super().__init__("RY", indices, (theta,), time)

    def _build_matrix(self) -> np.ndarray:
        c, s = np.cos(self.params[0]), np.sin(self.params[0])
        return np.array([[c, -s], [s, c]], dtype=complex)


class RZ(ParametricGate):
    """exp(-i θ Z) = diag(e^{-iθ}, e^{iθ})."""

    def __init__(
        self, indices: int | list[int], theta: float, time: float | None = None
    ):
        super().__init__("RZ", indices, (theta,), time)

    def _build_matrix(self) -> np.ndarray:
        t = self.params[0]
        return np.diag([np.exp(-1j * t), np.exp(1j * t)])


# ---------------------------------------------------------------------------
# Two-qubit MatrixGate subclasses
# All use (control, target) index ordering.
# ---------------------------------------------------------------------------


class CNOT(MatrixGate):
    """Controlled-X in (control, target) convention."""

    def __init__(self, indices: list[int], time: float | None = None):
        mat = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        super().__init__("CNOT", indices, mat, time)


class CZ(MatrixGate):
    def __init__(self, indices: list[int], time: float | None = None):
        super().__init__("CZ", indices, np.diag([1, 1, 1, -1]).astype(complex), time)


class SWAP(MatrixGate):
    def __init__(self, indices: list[int], time: float | None = None):
        mat = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        super().__init__("SWAP", indices, mat, time)


class iSWAP(MatrixGate):
    def __init__(self, indices: list[int], time: float | None = None):
        mat = np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        super().__init__("iSWAP", indices, mat, time)


# ---------------------------------------------------------------------------
# Two-qubit ParametricGate subclasses
# Convention: exp(-i θ P⊗P), full angle.
# ---------------------------------------------------------------------------


class RXX(ParametricGate):
    """exp(-i θ X⊗X) = cos(θ)I⊗I - i sin(θ) X⊗X."""

    def __init__(self, indices: list[int], theta: float, time: float | None = None):
        super().__init__("RXX", indices, (theta,), time)

    def _build_matrix(self) -> np.ndarray:
        c, s = np.cos(self.params[0]), np.sin(self.params[0])
        return np.array(
            [
                [c, 0, 0, -1j * s],
                [0, c, -1j * s, 0],
                [0, -1j * s, c, 0],
                [-1j * s, 0, 0, c],
            ],
        )


class RYY(ParametricGate):
    """exp(-i θ Y⊗Y) = cos(θ)I⊗I - i sin(θ) Y⊗Y."""

    def __init__(self, indices: list[int], theta: float, time: float | None = None):
        super().__init__("RYY", indices, (theta,), time)

    def _build_matrix(self) -> np.ndarray:
        c, s = np.cos(self.params[0]), np.sin(self.params[0])
        # Y⊗Y = [[0,0,0,-1],[0,0,1,0],[0,1,0,0],[-1,0,0,0]]
        return np.array(
            [
                [c, 0, 0, 1j * s],
                [0, c, -1j * s, 0],
                [0, -1j * s, c, 0],
                [1j * s, 0, 0, c],
            ],
        )


class RZZ(ParametricGate):
    """exp(-i θ Z⊗Z) = diag(e^{-iθ}, e^{iθ}, e^{iθ}, e^{-iθ})."""

    def __init__(self, indices: list[int], theta: float, time: float | None = None):
        super().__init__("RZZ", indices, (theta,), time)

    def _build_matrix(self) -> np.ndarray:
        t = self.params[0]
        return np.diag(
            [np.exp(-1j * t), np.exp(1j * t), np.exp(1j * t), np.exp(-1j * t)]
        )


# ---------------------------------------------------------------------------
# Qutrit MatrixGate subclasses (local_dim=3, inferred from matrix shape)
# ---------------------------------------------------------------------------


class X01(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__(
            "X01", indices, np.array([[0, -1j, 0], [-1j, 0, 0], [0, 0, 1]]), time
        )


class SX01(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        mat = (1 / np.sqrt(2)) * np.array(
            [[1, -1j, 0], [-1j, 1, 0], [0, 0, np.sqrt(2)]]
        )
        super().__init__("SX01", indices, mat, time)


class X12(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__(
            "X12", indices, np.array([[1, 0, 0], [0, 0, -1j], [0, -1j, 0]]), time
        )


class SX12(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        mat = (1 / np.sqrt(2)) * np.array(
            [[np.sqrt(2), 0, 0], [0, 1, -1j], [0, -1j, 1]]
        )
        super().__init__("SX12", indices, mat, time)


class H01(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        s = 1 / np.sqrt(2)
        mat = np.array([[s, s, 0], [s, -s, 0], [0, 0, 1]])
        super().__init__("H01", indices, mat, time)


class SDG01(MatrixGate):
    def __init__(self, indices: int | list[int], time: float | None = None):
        super().__init__(
            "SDG01", indices, np.array([[1, 0, 0], [0, -1j, 0], [0, 0, 1]]), time
        )


class SWAP3(MatrixGate):
    """Two-qutrit SWAP."""

    def __init__(self, indices: list[int], time: float | None = None):
        mat = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        super().__init__("SWAP3", indices, mat, time)


# CNOT3 matrix: legacy uses (target, control) ordering.
# v0.2 (control, target) matrix = SWAP3 @ M_legacy_final @ SWAP3.T
# where M_legacy_final is the literal from legacy/gates.py transposed.
_SWAP3_M = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)
_CNOT3_LEGACY_T = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1 / np.sqrt(2), 0.0, 0.0, -1 / np.sqrt(2), 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2), 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1j, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
)
_CNOT3_M = _SWAP3_M @ _CNOT3_LEGACY_T @ _SWAP3_M.T


class CNOT3(MatrixGate):
    """Two-qutrit CNOT in (control, target) convention."""

    def __init__(self, indices: list[int], time: float | None = None):
        super().__init__("CNOT3", indices, _CNOT3_M, time)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def random_unitary(n: int, local_dim: int = 2, seed: int | None = None) -> np.ndarray:
    """Haar-random d^n × d^n unitary. Uses np.random.default_rng(seed)."""
    rng = np.random.default_rng(seed)
    dim = local_dim**n
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    _, u = np.linalg.eigh(a @ a.conj().T)
    return u


def is_unitary(u: np.ndarray, atol: float = 1e-10) -> bool:
    n = u.shape[0]
    return np.allclose(u @ u.conj().T, np.eye(n), atol=atol) and np.allclose(
        u.conj().T @ u, np.eye(n), atol=atol
    )


def pauli_string_to_gates(string: str) -> list[Gate]:
    """'XZI' → [X(0), Z(1)] — identity sites skipped, case-insensitive."""
    _map = {"x": X, "y": Y, "z": Z}
    return [_map[op.lower()](i) for i, op in enumerate(string) if op.upper() != "I"]
