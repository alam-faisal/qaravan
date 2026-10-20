"""Physics-correctness tests for core/gates.py."""

import numpy as np
import pytest
from scipy.linalg import expm
from qaravan.core.base import (
    IncompatibleNoiseError,
    IncompatibleStateError,
    Simulator,
    State,
)
from qaravan.core.circuit import Circuit
from qaravan.core.gates import (
    CNOT,
    CNOT3,
    CRX,
    CZ,
    H01,
    RX,
    RXX,
    RY,
    RYY,
    RZ,
    RZZ,
    SDG01,
    SWAP,
    SWAP3,
    SX,
    SX01,
    SX12,
    X01,
    X12,
    XY,
    H,
    I,
    S,
    Sdg,
    T,
    Tdg,
    X,
    Y,
    Z,
    is_unitary,
    iSWAP,
    pauli_string_to_gates,
    random_unitary,
)

X_MATRIX = np.array([[0, 1], [1, 0]], dtype=complex)
Y_MATRIX = np.array([[0, -1j], [1j, 0]])
Z_MATRIX = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# Pauli matrices
# ---------------------------------------------------------------------------


def test_pauli_matrices():
    assert np.allclose(X(0).matrix, X_MATRIX)
    assert np.allclose(Y(0).matrix, Y_MATRIX)
    assert np.allclose(Z(0).matrix, Z_MATRIX)


# ---------------------------------------------------------------------------
# Single-qubit gate inverses
# ---------------------------------------------------------------------------


def test_hadamard_self_inverse():
    assert np.allclose(H(0).dagger().matrix, H(0).matrix)


def test_s_sdg_inverse():
    assert np.allclose(S(0).matrix @ Sdg(0).matrix, np.eye(2))


def test_t_tdg_inverse():
    assert np.allclose(T(0).matrix @ Tdg(0).matrix, np.eye(2))


# ---------------------------------------------------------------------------
# Parametric single-qubit limit cases
# Convention: exp(-i θ P) — full angle, no 1/2 factor.
# ---------------------------------------------------------------------------


def test_rx_at_zero_is_identity():
    assert np.allclose(RX(0, 0).matrix, np.eye(2))


def test_rx_at_half_pi_is_minus_ix():
    assert np.allclose(RX(0, np.pi / 2).matrix, -1j * X_MATRIX)


def test_rz_at_half_pi_is_minus_iz():
    assert np.allclose(RZ(0, np.pi / 2).matrix, -1j * Z_MATRIX)


def test_ry_at_zero_is_identity():
    assert np.allclose(RY(0, 0).matrix, np.eye(2))


# ---------------------------------------------------------------------------
# Two-qubit parametric gates
# ---------------------------------------------------------------------------


def test_rzz_diagonal():
    theta = 0.7
    diag = np.diag(RZZ([0, 1], theta).matrix)
    expected = np.array(
        [
            np.exp(-1j * theta),
            np.exp(1j * theta),
            np.exp(1j * theta),
            np.exp(-1j * theta),
        ]
    )
    assert np.allclose(diag, expected)


def test_rxx_limit():
    assert np.allclose(RXX([0, 1], 0).matrix, np.eye(4))


def test_ryy_limit():
    assert np.allclose(RYY([0, 1], 0).matrix, np.eye(4))


def test_xy_limit():
    assert np.allclose(XY([0, 1], 0).matrix, np.eye(4))


def test_xy_matches_expm():
    X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
    Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
    xx_plus_yy = np.kron(X_mat, X_mat) + np.kron(Y_mat, Y_mat)
    for theta in [0.3, 1.0, -0.7, 2.1]:
        exact = expm(-1j * theta * xx_plus_yy)
        assert np.allclose(XY([0, 1], theta).matrix, exact)


def test_xy_is_unitary():
    assert is_unitary(XY([0, 1], 0.42).matrix)


def test_xy_dagger():
    g = XY([0, 1], 0.5)
    gd = g.dagger()
    assert isinstance(gd, XY)
    assert np.allclose(gd.matrix, XY([0, 1], -0.5).matrix)


# ---------------------------------------------------------------------------
# CNOT convention: (control, target), big-endian qubit ordering
# ---------------------------------------------------------------------------


def test_cnot_convention():
    # |10⟩ = index 2 in big-endian → control=1 flips target → |11⟩ = index 3
    psi = np.zeros(4)
    psi[2] = 1.0
    assert np.allclose(CNOT([0, 1]).matrix @ psi, np.array([0, 0, 0, 1]))


def test_cnot_is_unitary():
    assert is_unitary(CNOT([0, 1]).matrix)


# ---------------------------------------------------------------------------
# SWAP self-inverse
# ---------------------------------------------------------------------------


def test_swap_self_inverse():
    assert np.allclose(SWAP([0, 1]).dagger().matrix, SWAP([0, 1]).matrix)


# ---------------------------------------------------------------------------
# pauli_string_to_gates
# ---------------------------------------------------------------------------


def test_pauli_string_to_gates_basic():
    gates = pauli_string_to_gates("XZI")
    assert len(gates) == 2
    assert gates[0].indices == [0]
    assert gates[1].indices == [1]
    assert np.allclose(gates[0].matrix, X_MATRIX)
    assert np.allclose(gates[1].matrix, Z_MATRIX)


def test_pauli_string_to_gates_case_insensitive():
    lower = pauli_string_to_gates("xzi")
    upper = pauli_string_to_gates("XZI")
    assert np.allclose(lower[0].matrix, upper[0].matrix)
    assert np.allclose(lower[1].matrix, upper[1].matrix)


# ---------------------------------------------------------------------------
# random_unitary
# ---------------------------------------------------------------------------


def test_random_unitary_is_unitary():
    assert is_unitary(random_unitary(2))


def test_random_unitary_seed_reproducible():
    u1 = random_unitary(2, seed=0)
    u2 = random_unitary(2, seed=0)
    assert np.allclose(u1, u2)


def test_random_unitary_different_seeds():
    u1 = random_unitary(2, seed=0)
    u2 = random_unitary(2, seed=1)
    assert not np.allclose(u1, u2)


def test_random_unitary_local_dim_3():
    u = random_unitary(1, local_dim=3, seed=7)
    assert u.shape == (3, 3)
    assert is_unitary(u)


# ---------------------------------------------------------------------------
# All non-parametric qubit gates are unitary
# ---------------------------------------------------------------------------

_QUBIT_GATES = [
    I(0),
    X(0),
    Y(0),
    Z(0),
    H(0),
    S(0),
    Sdg(0),
    T(0),
    Tdg(0),
    SX(0),
    CNOT([0, 1]),
    CZ([0, 1]),
    SWAP([0, 1]),
    iSWAP([0, 1]),
]


@pytest.mark.parametrize("gate", _QUBIT_GATES, ids=lambda g: g.name)
def test_all_qubit_gates_unitary(gate):
    assert is_unitary(gate.matrix)


def test_sx_squares_to_x():
    """Pins SX's phase: exp(-i pi X / 4) is also unitary and also squares to X up to phase."""
    assert np.allclose(SX(0).matrix @ SX(0).matrix, X(0).matrix)


# ---------------------------------------------------------------------------
# ParametricGate dagger roundtrip
# ---------------------------------------------------------------------------


def test_parametric_gate_dagger_roundtrip():
    g = RX(0, np.pi / 4)
    gd = g.dagger()
    assert isinstance(gd, RX)
    assert gd.params == (-np.pi / 4,)
    assert np.allclose(gd.matrix, RX(0, -np.pi / 4).matrix)


def test_parametric_gate_dagger_roundtrip_rzz():
    g = RZZ([0, 1], 0.5)
    gd = g.dagger()
    assert isinstance(gd, RZZ)
    assert np.allclose(gd.matrix, RZZ([0, 1], -0.5).matrix)


def test_parametric_gate_str_shows_params():
    s = str(RX(0, np.pi / 2))
    assert "RX" in s
    assert "1.571" in s


# ---------------------------------------------------------------------------
# Qutrit gates unitarity
# ---------------------------------------------------------------------------

_QUTRIT_GATES = [
    X01(0),
    SX01(0),
    X12(0),
    SX12(0),
    H01(0),
    SDG01(0),
    SWAP3([0, 1]),
    CNOT3([0, 1]),
]


@pytest.mark.parametrize("gate", _QUTRIT_GATES, ids=lambda g: g.name)
def test_qutrit_gates_unitary(gate):
    assert is_unitary(gate.matrix)


def test_swap3_local_dim():
    assert SWAP3([0, 1]).local_dim == 3


def test_cnot3_local_dim():
    assert CNOT3([0, 1]).local_dim == 3


# ---------------------------------------------------------------------------
# State.apply structural test (wires default_simulator)
# ---------------------------------------------------------------------------


class _MinState(State):
    @property
    def default_simulator(self):
        return _MinSim

    def expectation(self, o):
        return 0.0

    def sample(self, s):
        return np.array([])

    def measure_and_collapse(self, s):
        return (self, "0" * len(s))

    def overlap(self, o):
        return 1.0

    def __repr__(self):
        return "_MinState()"


class _MinSim(Simulator):
    def _validate_state(self, state):
        if not isinstance(state, _MinState):
            raise IncompatibleStateError

    def _validate_noise(self, noise_model):
        raise IncompatibleNoiseError

    def _insert_noise(self, circuit, noise_model):
        return circuit

    def translate_gate(self, gate):
        return gate.matrix

    def _apply_translated_gate(self, state, tg):
        pass


def test_state_apply():
    circ = Circuit([H(0), CNOT([0, 1])], num_sites=2)
    result = _MinState().apply(circ)
    assert isinstance(result, State)


# ---------------------------------------------------------------------------
# CRX
# ---------------------------------------------------------------------------


def test_crx_matches_exponential_of_generator():
    """exp(-i θ n_control ⊗ X_target), with n = (I - Z)/2."""
    theta = 0.7391
    number_op = (np.eye(2) - Z_MATRIX) / 2
    expected = expm(-1j * theta * np.kron(number_op, X_MATRIX))
    assert np.allclose(CRX([0, 1], theta).matrix, expected)


def test_crx_zero_angle_is_identity():
    assert np.allclose(CRX([0, 1], 0.0).matrix, np.eye(4))


def test_crx_dagger_negates_the_angle():
    theta = 0.31
    assert np.allclose(CRX([0, 1], theta).dagger().matrix, CRX([0, 1], -theta).matrix)


def test_crx_control_and_target_are_not_interchangeable():
    """Control on 1 / target on 0 is a different operator, and to_matrix must show it."""
    theta = 0.4517
    forward = Circuit([CRX([0, 1], theta)], num_sites=2).to_matrix()
    reversed_ = Circuit([CRX([1, 0], theta)], num_sites=2).to_matrix()
    assert not np.allclose(forward, reversed_)

    number_op = (np.eye(2) - Z_MATRIX) / 2
    assert np.allclose(reversed_, expm(-1j * theta * np.kron(X_MATRIX, number_op)))
