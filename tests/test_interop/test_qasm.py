"""Tests for qaravan.interop.qasm -- unitary-only QASM text -> Qaravan Circuit converter.

Fixtures below are hand-written QASM text, deliberately not sourced from
greens_functions_experiment/datacraft at runtime (see
proposals/2026-08-21-task24-qasm-consumer.md), but matching the exact grammar confirmed by
reading that repo's Magritte exporter (qasm_export.py) and its own test suite: header/qreg/
barrier line shapes, the exact `gate rzz(...)`/`gate xy(...)` block text, and the conditional
`// global_phase = <value>` comment. A few also cover symbolic-pi angle expressions
(`pi/2`, etc.), which the exporter itself never emits but other QASM sources (e.g. Qiskit) do.
"""

import numpy as np
import pytest
from qaravan.core.gates import (
    CNOT,
    CZ,
    RX,
    RY,
    RZ,
    RZZ,
    SX,
    XY,
    H,
    S,
    Sdg,
    T,
    Tdg,
    X,
    Y,
    Z,
)
from qaravan.interop.qasm import from_unitary_qasm

_HEADER = (
    "OPENQASM 2.0;\n"
    'include "qelib1.inc";\n'
    "// Angle convention: exp(-i*theta*P/2) for all parameterised Pauli rotations (rz, rzz, xy)\n"
)

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


@pytest.mark.parametrize("qasm_name,expected_cls", [("cx", CNOT), ("cz", CZ)])
def test_two_qubit_fixed_gates(qasm_name, expected_cls):
    qasm = _HEADER + f"qreg q[2];\n{qasm_name} q[0], q[1];\n"
    circuit = from_unitary_qasm(qasm, num_sites=2)
    assert len(circuit.gates) == 1
    result = circuit.gates[0]
    assert isinstance(result, expected_cls)
    assert result.indices == [0, 1]
    assert np.allclose(result.matrix, expected_cls([0, 1]).matrix)


def test_two_qubit_fixed_unsupported_raises():
    qasm = _HEADER + "qreg q[2];\nswap q[0], q[1];\n"
    with pytest.raises(NotImplementedError, match="swap"):
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
# End-to-end -- same "hopping bond + onsite energy" shape as
# test_magritte.py's own end-to-end test (task22 Section 2), via the xy custom
# block instead of the protobuf path
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
