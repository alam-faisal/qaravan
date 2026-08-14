"""Tests for qaravan.interop.magritte -- the Magritte gate-dict -> Qaravan Circuit converter.

No real Magritte export exists yet (see
proposals/2026-08-14-task22-magritte-circuit-converter.md); these tests build
hand-constructed Gate protobuf messages, round-tripped through the same MessageToDict
call greens_functions_experiment's export_hardware_layering uses, rather than parsing an
actual exported file.
"""

import numpy as np
import pytest
from google.protobuf.json_format import MessageToDict

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
    S,
    Sdg,
    T,
    Tdg,
    X,
    Y,
    Z,
    iSWAP,
)
from qaravan.interop.magritte import _gate_from_proto, from_magritte_json
from qaravan.interop.proto import hardware_gates_pb2 as hg


def _dict(gate: hg.Gate) -> dict:
    """Round-trip through MessageToDict -- the exact wire shape (camelCase keys)
    export_hardware_layering produces, not hand-guessed JSON."""
    return MessageToDict(gate)


# ---------------------------------------------------------------------------
# One-qubit fixed gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixed_enum,expected_cls",
    [
        (hg.H, H),
        (hg.X_GATE, X),
        (hg.Y_GATE, Y),
        (hg.Z_GATE, Z),
        (hg.S, S),
        (hg.Sdg, Sdg),
        (hg.T, T),
        (hg.Tdg, Tdg),
        (hg.SX, SX),
    ],
)
def test_one_qubit_fixed_gates(fixed_enum, expected_cls):
    gate = hg.Gate(qubits=["3"])
    gate.one_qubit_gate.fixed = fixed_enum
    result = _gate_from_proto(gate)
    assert isinstance(result, expected_cls)
    assert result.indices == [3]
    assert np.allclose(result.matrix, expected_cls(3).matrix)


def test_one_qubit_fixed_unsupported_raises():
    gate = hg.Gate(qubits=["0"])
    gate.one_qubit_gate.fixed = hg.SXdg
    with pytest.raises(NotImplementedError, match="SXdg"):
        _gate_from_proto(gate)


# ---------------------------------------------------------------------------
# One-qubit parameterised gates -- angle halving
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param_enum,expected_cls", [(hg.Rx, RX), (hg.Ry, RY), (hg.Rz, RZ)]
)
def test_one_qubit_parameterised_gates_halve_angle(param_enum, expected_cls):
    gate = hg.Gate(qubits=["2"], angle=1.4)
    gate.one_qubit_gate.parameterised = param_enum
    result = _gate_from_proto(gate)
    assert isinstance(result, expected_cls)
    assert np.allclose(result.matrix, expected_cls(2, 0.7).matrix)


def test_one_qubit_parameterised_unsupported_raises():
    gate = hg.Gate(qubits=["0"], angle=0.5)
    gate.one_qubit_gate.parameterised = hg.U3
    with pytest.raises(NotImplementedError, match="U3"):
        _gate_from_proto(gate)


# ---------------------------------------------------------------------------
# Two-qubit fixed gates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixed_enum,expected_cls",
    [(hg.CZ, CZ), (hg.ISWAP, iSWAP), (hg.CX, CNOT), (hg.SWAP, SWAP)],
)
def test_two_qubit_fixed_gates(fixed_enum, expected_cls):
    gate = hg.Gate(qubits=["1", "4"])
    gate.two_qubit_gate.fixed = fixed_enum
    result = _gate_from_proto(gate)
    assert isinstance(result, expected_cls)
    assert result.indices == [1, 4]
    assert np.allclose(result.matrix, expected_cls([1, 4]).matrix)


def test_two_qubit_fixed_unsupported_raises():
    gate = hg.Gate(qubits=["0", "1"])
    gate.two_qubit_gate.fixed = hg.ECR
    with pytest.raises(NotImplementedError, match="ECR"):
        _gate_from_proto(gate)


# ---------------------------------------------------------------------------
# Two-qubit parameterised gates -- angle halving, including the new XY gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "param_enum,expected_cls",
    [(hg.RXX, RXX), (hg.RYY, RYY), (hg.RZZ, RZZ), (hg.XY, XY)],
)
def test_two_qubit_parameterised_gates_halve_angle(param_enum, expected_cls):
    gate = hg.Gate(qubits=["0", "1"], angle=0.8)
    gate.two_qubit_gate.parameterised = param_enum
    result = _gate_from_proto(gate)
    assert isinstance(result, expected_cls)
    assert np.allclose(result.matrix, expected_cls([0, 1], 0.4).matrix)


def test_two_qubit_parameterised_unsupported_raises():
    gate = hg.Gate(qubits=["0", "1"], angle=0.5)
    gate.two_qubit_gate.parameterised = hg.MS
    with pytest.raises(NotImplementedError, match="MS"):
        _gate_from_proto(gate)


# ---------------------------------------------------------------------------
# Global phase
# ---------------------------------------------------------------------------


def test_global_phase():
    gate = hg.Gate(zero_qubit_gate=hg.GLOBAL_PHASE, phase=-0.375)
    result = _gate_from_proto(gate)
    assert result.indices == [0]
    assert np.allclose(result.matrix, np.exp(-0.375j) * np.eye(2))


def test_zero_qubit_gate_unspecified_raises():
    # Explicitly assigning the enum's zero value still registers as "set" for
    # WhichOneof -- confirmed directly against the generated message -- so this must
    # be rejected rather than silently treated as "no zero_qubit_gate at all".
    gate = hg.Gate()
    gate.zero_qubit_gate = hg.ZERO_QUBIT_GATE_UNSPECIFIED
    with pytest.raises(NotImplementedError):
        _gate_from_proto(gate)


def test_gate_with_no_gate_type_set_raises():
    gate = hg.Gate(qubits=["0"])
    with pytest.raises(NotImplementedError, match="no gate_type"):
        _gate_from_proto(gate)


# ---------------------------------------------------------------------------
# from_magritte_json -- realistic small circuit, from real MessageToDict output
# ---------------------------------------------------------------------------


def test_from_magritte_json_end_to_end_matches_hand_built_circuit():
    """A tiny 'hopping bond + onsite energy' pattern -- exactly AIM's own compile shape
    per task22 Section 2 (weight-2 term -> XY, weight-1 diagonal term -> Rz)."""
    hop = hg.Gate(qubits=["0", "1"], angle=0.6)
    hop.two_qubit_gate.parameterised = hg.XY
    onsite0 = hg.Gate(qubits=["0"], angle=0.3)
    onsite0.one_qubit_gate.parameterised = hg.Rz
    onsite1 = hg.Gate(qubits=["1"], angle=-0.2)
    onsite1.one_qubit_gate.parameterised = hg.Rz

    gate_dicts = [_dict(hop), _dict(onsite0), _dict(onsite1)]
    circuit = from_magritte_json(gate_dicts, num_sites=2)

    expected = [XY([0, 1], 0.3), RZ(0, 0.15), RZ(1, -0.1)]
    assert circuit.num_sites == 2
    assert len(circuit.gates) == 3
    for actual_gate, expected_gate in zip(circuit.gates, expected):
        assert type(actual_gate) is type(expected_gate)
        assert np.allclose(actual_gate.matrix, expected_gate.matrix)


def test_from_magritte_json_preserves_order():
    gates_in = []
    for q, angle, param in [(0, 0.2, hg.Rz), (1, 0.4, hg.Rx), (0, 0.6, hg.Ry)]:
        g = hg.Gate(qubits=[str(q)], angle=angle)
        g.one_qubit_gate.parameterised = param
        gates_in.append(g)
    gate_dicts = [_dict(g) for g in gates_in]
    circuit = from_magritte_json(gate_dicts, num_sites=2)
    assert [type(g).__name__ for g in circuit.gates] == ["RZ", "RX", "RY"]
