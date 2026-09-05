"""Parse unitary-only OpenQASM 2 text into a Qaravan Circuit -- a text-based sibling to
interop/magritte.py's protobuf/JSON path, not a general-purpose QASM parser. Built for the
dialect Magritte's own QASM exporter emits (h, x, y, z, s, sdg, t, tdg, sx, cx, cz, rx, ry, rz,
rzz, xy, plus its custom rzz/xy gate blocks and global-phase comment), but that dialect isn't
Magritte-exclusive: any unitary-only QASM2 using a `q`-named register and this gate vocabulary
parses the same way -- Qaravan has no classical-register/measurement concept in this pipeline,
so `creg`/`measure` are permanently out of scope, not a gap. See
proposals/2026-08-21-task24-qasm-consumer.md for the design rationale.
"""

from __future__ import annotations
import ast
import operator
import re
from collections.abc import Iterator
import numpy as np
from qaravan.core.circuit import Circuit
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
    MatrixGate,
    S,
    Sdg,
    T,
    Tdg,
    X,
    Y,
    Z,
)

_ONE_QUBIT_FIXED = {
    "h": H,
    "x": X,
    "y": Y,
    "z": Z,
    "s": S,
    "sdg": Sdg,
    "t": T,
    "tdg": Tdg,
    "sx": SX,
}
_TWO_QUBIT_FIXED = {"cx": CNOT, "cz": CZ}
_ONE_QUBIT_PARAMETERISED = {"rx": RX, "ry": RY, "rz": RZ}
_TWO_QUBIT_PARAMETERISED = {"rzz": RZZ, "xy": XY}

_GLOBAL_PHASE_RE = re.compile(r"^//\s*global_phase\s*=\s*(.+)$")
_QREG_RE = re.compile(r"^qreg\s+q\[(\d+)\]\s*;$")
_QUBIT_RE = re.compile(r"q\[(\d+)\]")
_INSTRUCTION_RE = re.compile(
    r"^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\((?P<angle>[^)]*)\))?"
    r"\s+(?P<qubits>q\[\d+\](?:\s*,\s*q\[\d+\])*)\s*;$"
)


_ANGLE_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_ANGLE_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_angle_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == "pi":
        return np.pi
    if isinstance(node, ast.BinOp) and type(node.op) in _ANGLE_BINOPS:
        return _ANGLE_BINOPS[type(node.op)](
            _eval_angle_node(node.left), _eval_angle_node(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ANGLE_UNARYOPS:
        return _ANGLE_UNARYOPS[type(node.op)](_eval_angle_node(node.operand))
    raise NotImplementedError(
        f"Unsupported QASM angle expression node: {ast.dump(node)}"
    )


def _parse_angle(text: str) -> float:
    """Evaluate a QASM angle expression: numeric literals, `pi`, +-*/^ and parentheses --
    not the full OpenQASM expression grammar (no sin/cos/sqrt/etc.), but enough for the
    symbolic-pi expressions real compilers (Qiskit included) actually emit. Parsed via `ast`,
    never `eval`, so a malformed or hostile angle string can't execute anything -- it just
    fails to match the allowed node types above.
    """
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError as exc:
        raise NotImplementedError(
            f"Could not parse QASM angle expression {text!r}"
        ) from exc
    return _eval_angle_node(tree)


def _unimplemented(name: str) -> NotImplementedError:
    return NotImplementedError(
        f"QASM gate '{name}' has no Qaravan equivalent yet -- not part of the gate "
        "vocabulary this importer supports; add it if a Hamiltonian/gateset choice needs it."
    )


def _parse_qubits(text: str) -> list[int]:
    return [int(match.group(1)) for match in _QUBIT_RE.finditer(text)]


def _gate_from_instruction(name: str, angle: str | None, qubits: list[int]):
    """One parsed (gate_name, angle, qubits) triple -> one Qaravan Gate.

    angle is Magritte's own half-angle theta (raw QASM parameter text, or None for a fixed
    gate); halved here to match Qaravan's full-angle convention, exactly interop/magritte.py's
    `gate.angle / 2`.
    """
    if angle is None:
        table = _ONE_QUBIT_FIXED if len(qubits) == 1 else _TWO_QUBIT_FIXED
        if name not in table:
            raise _unimplemented(name)
        return table[name](qubits[0] if len(qubits) == 1 else qubits)
    theta = _parse_angle(angle) / 2
    table = _ONE_QUBIT_PARAMETERISED if len(qubits) == 1 else _TWO_QUBIT_PARAMETERISED
    if name not in table:
        raise _unimplemented(name)
    return table[name](qubits[0] if len(qubits) == 1 else qubits, theta)


def _check_qreg(line: str, num_sites: int) -> None:
    match = _QREG_RE.match(line)
    if not match:
        raise NotImplementedError(f"Could not parse qreg declaration: {line!r}")
    declared = int(match.group(1))
    if declared != num_sites:
        raise ValueError(f"QASM declares qreg q[{declared}] but num_sites={num_sites}.")


def _skip_gate_block(first_line: str, lines: Iterator[str]) -> None:
    """Skip a `gate <name>(...) ... { ... }` declaration by brace depth, without ever
    interpreting its body -- rzz/xy's blocks are recognized and dispatched by invocation
    name alone (see module docstring)."""
    depth = first_line.count("{") - first_line.count("}")
    while depth > 0:
        next_line = next(lines)
        depth += next_line.count("{") - next_line.count("}")


def from_unitary_qasm(qasm_text: str, num_sites: int) -> Circuit:
    """Parse unitary-only QASM text (see module docstring) into a Qaravan Circuit.

    Recognizes header/qreg/barrier lines, an unparsed angle-convention comment, an optional
    `// global_phase = <value>` comment (absent means phase 0), and `gate rzz(...)`/
    `gate xy(...)` custom blocks. Raises NotImplementedError by name for any other gate,
    matching interop/magritte.py's convention.
    """
    gates = []
    global_phase = 0.0
    lines = iter(qasm_text.splitlines())
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        phase_match = _GLOBAL_PHASE_RE.match(stripped)
        if phase_match:
            global_phase += float(phase_match.group(1))
            continue
        if stripped.startswith("//"):
            continue
        if stripped.startswith("OPENQASM") or stripped.startswith("include"):
            continue
        if stripped.startswith("qreg"):
            _check_qreg(stripped, num_sites)
            continue
        if stripped.startswith("barrier"):
            continue
        if stripped.startswith("gate "):
            _skip_gate_block(stripped, lines)
            continue
        match = _INSTRUCTION_RE.match(stripped)
        if not match:
            raise NotImplementedError(f"Could not parse QASM line: {line!r}")
        qubits = _parse_qubits(match.group("qubits"))
        gates.append(
            _gate_from_instruction(match.group("name"), match.group("angle"), qubits)
        )

    if abs(global_phase) > 0:
        gates.insert(
            0, MatrixGate("GlobalPhase", [0], np.exp(1j * global_phase) * np.eye(2))
        )
    return Circuit(gates, num_sites=num_sites)
