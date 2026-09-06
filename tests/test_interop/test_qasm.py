"""Tests for qaravan.interop.qasm -- QASM text <-> Circuit.

Fixtures are hand-written QASM covering the grammar real exporters emit: header/qreg/barrier
lines, `gate rzz(...)`/`gate xy(...)` blocks, the conditional `// global_phase = <value>`
comment, and symbolic-pi angle expressions.
"""

import numpy as np
import pytest
from scipy.linalg import expm
from qaravan.core.circuit import Circuit
from qaravan.core.gates import (
    CNOT,
    CZ,
    RX,
    RXX,
    RY,
    RYY,
    RZ,
    RZZ,
    SWAP,
    SX,
    XY,
    H,
    I,
    MatrixGate,
    S,
    Sdg,
    T,
    Tdg,
    X,
    Y,
    Z,
    iSWAP,
)
from qaravan.interop.qasm import from_unitary_qasm, to_unitary_qasm

_HEADER = (
    "OPENQASM 2.0;\n"
    'include "qelib1.inc";\n'
    "// Angle convention: exp(-i*theta*P/2) for all parameterised Pauli rotations (rz, rzz, xy)\n"
)

_X_MAT = np.array([[0, 1], [1, 0]], dtype=complex)
_Y_MAT = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z_MAT = np.array([[1, 0], [0, -1]], dtype=complex)

_RZZ_BLOCK = "gate rzz(theta) a, b { cx a, b; rz(theta) b; cx a, b; }\n"
_XY_BLOCK = (
    "gate xy(theta) a, b {\n"
    "    rz(-pi/2) a; sx a; rz(pi/2) a; s b;\n"
    "    cx a, b; ry(-theta) a; ry(-theta) b; cx a, b;\n"
    "    sdg b; rz(-pi/2) a; sxdg a; rz(pi/2) a;\n"
    "}\n"
)


# ---------------------------------------------------------------------------
# One-qubit fixed gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qasm_name,expected_cls",
    [
        ("id", I),
        ("h", H),
        ("x", X),
        ("y", Y),
        ("z", Z),
        ("s", S),
        ("sdg", Sdg),
        ("t", T),
        ("tdg", Tdg),
        ("sx", SX),
    ],
)
def test_one_qubit_fixed_gates(qasm_name, expected_cls):
    qasm = _HEADER + f"qreg q[1];\n{qasm_name} q[0];\n"
    circuit = from_unitary_qasm(qasm, num_sites=1)
    assert len(circuit.gates) == 1
    result = circuit.gates[0]
    assert isinstance(result, expected_cls)
    assert result.indices == [0]
    assert np.allclose(result.matrix, expected_cls(0).matrix)


def test_one_qubit_fixed_unsupported_raises():
    qasm = _HEADER + "qreg q[1];\nsxdg q[0];\n"
    with pytest.raises(NotImplementedError, match="sxdg"):
        from_unitary_qasm(qasm, num_sites=1)


# ---------------------------------------------------------------------------
# Two-qubit fixed gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "qasm_name,expected_cls",
    [("cx", CNOT), ("cz", CZ), ("swap", SWAP), ("iswap", iSWAP)],
)
def test_two_qubit_fixed_gates(qasm_name, expected_cls):
    qasm = _HEADER + f"qreg q[2];\n{qasm_name} q[0], q[1];\n"
    circuit = from_unitary_qasm(qasm, num_sites=2)
    assert len(circuit.gates) == 1
    result = circuit.gates[0]
    assert isinstance(result, expected_cls)
    assert result.indices == [0, 1]
    assert np.allclose(result.matrix, expected_cls([0, 1]).matrix)


def test_two_qubit_fixed_unsupported_raises():
    qasm = _HEADER + "qreg q[2];\nch q[0], q[1];\n"
    with pytest.raises(NotImplementedError, match="ch"):
        from_unitary_qasm(qasm, num_sites=2)


# ---------------------------------------------------------------------------
# One-qubit parameterised gates -- angle halving
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("qasm_name,expected_cls", [("rx", RX), ("ry", RY), ("rz", RZ)])
def test_one_qubit_parameterised_gates_halve_angle(qasm_name, expected_cls):
    qasm = _HEADER + f"qreg q[1];\n{qasm_name}(0.8) q[0];\n"
    circuit = from_unitary_qasm(qasm, num_sites=1)
    result = circuit.gates[0]
    assert isinstance(result, expected_cls)
    assert np.allclose(result.matrix, expected_cls(0, 0.4).matrix)


@pytest.mark.parametrize(
    "qasm_name,expected_cls,generator",
    [
        ("rxx", RXX, np.kron(_X_MAT, _X_MAT)),
        ("ryy", RYY, np.kron(_Y_MAT, _Y_MAT)),
        ("rzz", RZZ, np.kron(_Z_MAT, _Z_MAT)),
    ],
)
def test_two_qubit_rotations_match_qiskit_convention(
    qasm_name, expected_cls, generator
):
    """Anchors against exp(-i theta P / 2) itself, not against Qaravan's own class."""
    theta = 0.7391
    qasm = _HEADER + _RZZ_BLOCK + f"qreg q[2];\n{qasm_name}({theta}) q[0], q[1];\n"
    circuit = from_unitary_qasm(qasm, num_sites=2)
    assert isinstance(circuit.gates[0], expected_cls)
    assert np.allclose(circuit.to_matrix(), expm(-0.5j * theta * generator))


# ---------------------------------------------------------------------------
# Angle expressions -- symbolic pi, as emitted by e.g. Qiskit (not the exporter
# itself, which always writes a plain decimal, but a QASM angle position can
# legally hold any arithmetic expression over `pi`)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected_theta",
    [
        ("pi", np.pi / 2),
        ("-pi/2", -np.pi / 4),
        ("pi/4", np.pi / 8),
        ("2*pi", np.pi),
        ("pi/2 + 0.1", (np.pi / 2 + 0.1) / 2),
    ],
)
def test_symbolic_pi_angle_expressions(expr, expected_theta):
    qasm = _HEADER + f"qreg q[1];\nrz({expr}) q[0];\n"
    circuit = from_unitary_qasm(qasm, num_sites=1)
    assert np.allclose(circuit.gates[0].matrix, RZ(0, expected_theta).matrix)


def test_unsupported_angle_expression_raises():
    # A bare unknown identifier -- reaches _parse_angle/_eval_angle_node itself, unlike a
    # function call (e.g. `sin(pi/4)`), whose nested parens never match the flat
    # single-paren-group instruction regex in the first place and fail one step earlier.
    qasm = _HEADER + "qreg q[1];\nrz(sqrt2) q[0];\n"
    with pytest.raises(NotImplementedError, match="angle expression"):
        from_unitary_qasm(qasm, num_sites=1)


# ---------------------------------------------------------------------------
# Two-qubit parameterised gates -- custom `gate` blocks, skipped by name and
# never semantically parsed, angle halving applied to the invocation only
# ---------------------------------------------------------------------------


def test_rzz_custom_block_is_skipped_and_dispatched_by_name():
    qasm = _HEADER + _RZZ_BLOCK + "qreg q[2];\nrzz(0.8) q[0], q[1];\n"
    circuit = from_unitary_qasm(qasm, num_sites=2)
    assert len(circuit.gates) == 1
    result = circuit.gates[0]
    assert isinstance(result, RZZ)
    assert np.allclose(result.matrix, RZZ([0, 1], 0.4).matrix)


def test_xy_custom_block_is_skipped_and_dispatched_by_name():
    qasm = _HEADER + _XY_BLOCK + "qreg q[2];\nxy(0.6) q[0], q[1];\n"
    circuit = from_unitary_qasm(qasm, num_sites=2)
    assert len(circuit.gates) == 1
    result = circuit.gates[0]
    assert isinstance(result, XY)
    assert np.allclose(result.matrix, XY([0, 1], 0.3).matrix)


def test_both_custom_blocks_present_before_qreg():
    """Realistic exporter shape: both blocks declared (each only when actually used), in the
    header, before qreg -- confirms block-skipping doesn't consume past its own closing brace."""
    qasm = (
        _HEADER
        + _RZZ_BLOCK
        + _XY_BLOCK
        + "qreg q[2];\nxy(0.6) q[0], q[1];\nrzz(0.8) q[0], q[1];\n"
    )
    circuit = from_unitary_qasm(qasm, num_sites=2)
    assert [type(g).__name__ for g in circuit.gates] == ["XY", "RZZ"]


# ---------------------------------------------------------------------------
# Global phase -- conditional comment
# ---------------------------------------------------------------------------


def test_global_phase_present_is_prepended():
    qasm = _HEADER + "// global_phase = 0.375\nqreg q[1];\nh q[0];\n"
    circuit = from_unitary_qasm(qasm, num_sites=1)
    assert len(circuit.gates) == 2
    assert circuit.gates[0].indices == [0]
    assert np.allclose(circuit.gates[0].matrix, np.exp(1j * 0.375) * np.eye(2))
    assert isinstance(circuit.gates[1], H)


def test_global_phase_absent_adds_no_gate():
    qasm = _HEADER + "qreg q[1];\nh q[0];\n"
    circuit = from_unitary_qasm(qasm, num_sites=1)
    assert len(circuit.gates) == 1
    assert isinstance(circuit.gates[0], H)


# ---------------------------------------------------------------------------
# barrier lines and multi-layer ordering
# ---------------------------------------------------------------------------


def test_barrier_lines_are_ignored_and_order_preserved():
    qasm = (
        _HEADER
        + "qreg q[2];\n"
        + "h q[0];\n"
        + "cx q[0], q[1];\n"
        + "barrier q[0],q[1];\n"
        + "rz(0.4) q[1];\n"
    )
    circuit = from_unitary_qasm(qasm, num_sites=2)
    assert [type(g).__name__ for g in circuit.gates] == ["H", "CNOT", "RZ"]
    assert np.allclose(circuit.gates[2].matrix, RZ(1, 0.2).matrix)


# ---------------------------------------------------------------------------
# qreg / num_sites consistency
# ---------------------------------------------------------------------------


def test_qreg_mismatch_raises():
    qasm = _HEADER + "qreg q[3];\nh q[0];\n"
    with pytest.raises(ValueError, match="num_sites"):
        from_unitary_qasm(qasm, num_sites=2)


# ---------------------------------------------------------------------------
# End-to-end -- a hopping bond plus onsite energies, through the xy custom block
# ---------------------------------------------------------------------------


def test_from_unitary_qasm_end_to_end_matches_hand_built_circuit():
    qasm = (
        _HEADER
        + _XY_BLOCK
        + "qreg q[2];\n"
        + "xy(0.6) q[0], q[1];\n"
        + "rz(0.3) q[0];\n"
        + "rz(-0.2) q[1];\n"
    )
    circuit = from_unitary_qasm(qasm, num_sites=2)

    expected = [XY([0, 1], 0.3), RZ(0, 0.15), RZ(1, -0.1)]
    assert circuit.num_sites == 2
    assert len(circuit.gates) == 3
    for actual_gate, expected_gate in zip(circuit.gates, expected):
        assert type(actual_gate) is type(expected_gate)
        assert np.allclose(actual_gate.matrix, expected_gate.matrix)


# ---------------------------------------------------------------------------
# Export -- to_unitary_qasm
# ---------------------------------------------------------------------------


def _every_supported_gate() -> Circuit:
    gates = [
        I(0),
        H(1),
        X(2),
        Y(0),
        Z(1),
        S(2),
        Sdg(0),
        T(1),
        Tdg(2),
        SX(0),
        CNOT([0, 1]),
        CZ([1, 2]),
        SWAP([0, 2]),
        iSWAP([1, 2]),
        RX(0, 0.3123),
        RY(1, -0.7311),
        RZ(2, 1.1907),
        RXX([0, 1], 0.4471),
        RYY([1, 2], -0.2213),
        RZZ([0, 2], 0.9137),
        XY([1, 2], 0.6029),
    ]
    return Circuit(gates, num_sites=3)


def test_round_trip_preserves_the_unitary():
    circuit = _every_supported_gate()
    back = from_unitary_qasm(to_unitary_qasm(circuit), num_sites=3)
    assert len(back.gates) == len(circuit.gates)
    assert np.allclose(back.to_matrix(), circuit.to_matrix())


def test_round_trip_preserves_angles_bit_exactly():
    """repr() round-trips float64; a %.6f style would silently lose the low bits."""
    rng = np.random.default_rng(0)
    angles = rng.uniform(-np.pi, np.pi, size=8)
    circuit = Circuit([RZ(0, angle) for angle in angles], num_sites=1)
    back = from_unitary_qasm(to_unitary_qasm(circuit), num_sites=1)
    assert [gate.params[0] for gate in back.gates] == list(angles)


def test_export_doubles_the_angle():
    """Pins the convention. A round trip alone cannot: doubling then halving always cancels."""
    assert "rx(0.6)" in to_unitary_qasm(Circuit([RX(0, 0.3)], num_sites=1))


def test_global_phase_round_trips():
    phase = 0.7
    phase_gate = MatrixGate("GlobalPhase", [0], np.exp(1j * phase) * np.eye(2))
    circuit = Circuit([phase_gate, H(0)], num_sites=1)
    text = to_unitary_qasm(circuit)
    assert "// global_phase = 0.7" in text
    assert np.allclose(
        from_unitary_qasm(text, num_sites=1).to_matrix(), circuit.to_matrix()
    )


def test_no_global_phase_comment_when_absent():
    assert "global_phase" not in to_unitary_qasm(Circuit([H(0)], num_sites=1))


def test_declarations_only_for_gates_used():
    text = to_unitary_qasm(Circuit([H(0), CNOT([0, 1])], num_sites=2))
    assert "gate " not in text  # h and cx are both in qelib1
    text = to_unitary_qasm(Circuit([RZZ([0, 1], 0.4)], num_sites=2))
    assert "gate rzz(theta) a, b { }" in text
    assert "gate xy" not in text


def test_export_rejects_qutrit_circuits():
    circuit = Circuit([H(0)], num_sites=1, local_dim=3)
    with pytest.raises(NotImplementedError, match="local_dim"):
        to_unitary_qasm(circuit)


def test_export_rejects_unsupported_gate():
    circuit = Circuit([MatrixGate("Custom", [0], np.eye(2))], num_sites=1)
    with pytest.raises(NotImplementedError, match="Custom"):
        to_unitary_qasm(circuit)
