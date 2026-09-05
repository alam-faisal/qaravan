"""Convert a compiled Magritte gate sequence (as JSON-able dicts) into a Qaravan Circuit.

See proposals/2026-08-14-task22-magritte-circuit-converter.md for the design rationale:
the rotation-angle-halving convention, which gate types AndersonImpurity's compile actually
produces, and why Qaravan vendors the schema (core/interop/proto/) instead of depending on
datacraft. This module imports only that vendored proto plus `protobuf` -- never `datacraft`.
"""

from __future__ import annotations
import numpy as np
from google.protobuf.json_format import ParseDict
from qaravan.core.base import Gate as QaravanGate
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
from qaravan.interop.proto import hardware_gates_pb2 as hg

# Every parameterised gate below uses Magritte's half-angle convention,
# exp(-i*angle/2*P); Qaravan's is the full-angle exp(-i*theta*P). Halving (gate.angle / 2,
# below) is uniform across every entry in both _..._PARAMETERISED tables -- not a per-gate
# special case -- since both sides use the same generator-exponential structure, just
# normalized a factor of 2 apart (task22 Section 1).

_ONE_QUBIT_FIXED = {
    hg.H: H,
    hg.X_GATE: X,
    hg.Y_GATE: Y,
    hg.Z_GATE: Z,
    hg.S: S,
    hg.Sdg: Sdg,
    hg.T: T,
    hg.Tdg: Tdg,
    hg.SX: SX,
}
_ONE_QUBIT_PARAMETERISED = {hg.Rx: RX, hg.Ry: RY, hg.Rz: RZ}
_TWO_QUBIT_FIXED = {hg.CZ: CZ, hg.ISWAP: iSWAP, hg.CX: CNOT, hg.SWAP: SWAP}
_TWO_QUBIT_PARAMETERISED = {hg.RXX: RXX, hg.RYY: RYY, hg.RZZ: RZZ, hg.XY: XY}


def _unimplemented(enum_type, value: int, category: str) -> NotImplementedError:
    name = enum_type.Name(value)
    return NotImplementedError(
        f"{category}.{name} has no Qaravan equivalent yet -- not exercised by "
        "AndersonImpurity's own compile under the star/chain geometry (see task22 "
        "proposal Section 2's table); add it if a Hamiltonian/gateset choice needs it."
    )


def _gate_from_proto(gate: hg.Gate) -> QaravanGate:
    """One parsed Magritte Gate message -> one Qaravan Gate.

    Raises NotImplementedError (by name) for anything not in the mapping tables above,
    rather than guessing at an unverified translation.
    """
    qubits = [int(q) for q in gate.qubits]
    kind = gate.WhichOneof("gate_type")

    if kind == "zero_qubit_gate":
        if gate.zero_qubit_gate == hg.GLOBAL_PHASE:
            # No qubits field at all on a GLOBAL_PHASE gate; physically a global phase
            # regardless of which site it's nominally attached to, so site 0 is arbitrary.
            return MatrixGate("GlobalPhase", [0], np.exp(1j * gate.phase) * np.eye(2))
        raise _unimplemented(hg.ZeroQubitGate, gate.zero_qubit_gate, "ZeroQubitGate")

    if kind == "one_qubit_gate":
        sub = gate.one_qubit_gate
        if sub.WhichOneof("gate") == "fixed":
            if sub.fixed not in _ONE_QUBIT_FIXED:
                raise _unimplemented(
                    hg.OneQubitGateFixed, sub.fixed, "OneQubitGateFixed"
                )
            return _ONE_QUBIT_FIXED[sub.fixed](qubits[0])
        if sub.parameterised not in _ONE_QUBIT_PARAMETERISED:
            raise _unimplemented(
                hg.OneQubitGateParameterised,
                sub.parameterised,
                "OneQubitGateParameterised",
            )
        return _ONE_QUBIT_PARAMETERISED[sub.parameterised](qubits[0], gate.angle / 2)

    if kind == "two_qubit_gate":
        sub = gate.two_qubit_gate
        if sub.WhichOneof("gate") == "fixed":
            if sub.fixed not in _TWO_QUBIT_FIXED:
                raise _unimplemented(
                    hg.TwoQubitGateFixed, sub.fixed, "TwoQubitGateFixed"
                )
            return _TWO_QUBIT_FIXED[sub.fixed](qubits)
        if sub.parameterised not in _TWO_QUBIT_PARAMETERISED:
            raise _unimplemented(
                hg.TwoQubitGateParameterised,
                sub.parameterised,
                "TwoQubitGateParameterised",
            )
        return _TWO_QUBIT_PARAMETERISED[sub.parameterised](qubits, gate.angle / 2)

    raise NotImplementedError(
        f"Gate has no gate_type set (WhichOneof returned {kind!r})."
    )


def from_magritte_json(gate_dicts: list[dict], num_sites: int) -> Circuit:
    """Parse a flat list of Magritte Gate-as-dict (e.g. from
    greens_functions_experiment's export_hardware_layering) into a Qaravan Circuit.

    Order is preserved exactly as given -- a Circuit is already just a flat ordered gate
    list, and the export is documented to already be flattened in application order
    (layer order, then term order within layer, then gate order within term).
    """
    gates = [_gate_from_proto(ParseDict(d, hg.Gate())) for d in gate_dicts]
    return Circuit(gates, num_sites=num_sites)
