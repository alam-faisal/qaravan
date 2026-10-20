"""Parse unitary-only OpenQASM 2 text into a Circuit.

Angles follow Qiskit's convention: a rotation named for generator P is exp(-i theta P / 2).
Qaravan's rotations are full-angle, exp(-i theta P), so every angle is halved on import.

Requires a `q`-named register. `creg`/`measure` are out of scope -- this pipeline has no
classical-register concept.
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

# === Import ===


_ONE_QUBIT_FIXED = {
    "id": I,
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
_TWO_QUBIT_FIXED = {"cx": CNOT, "cz": CZ, "swap": SWAP, "iswap": iSWAP}
_ONE_QUBIT_PARAMETERISED = {"rx": RX, "ry": RY, "rz": RZ}
_TWO_QUBIT_PARAMETERISED = {"rxx": RXX, "ryy": RYY, "rzz": RZZ, "xy": XY}

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
    """Evaluate a QASM angle expression: numeric literals, `pi`, + - * / ** and parentheses.

    Parsed via `ast`, never `eval`, so only those node types can appear; sin/cos/sqrt raise.
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
    """One parsed (gate_name, angle, qubits) triple -> one Gate.

    angle is the raw QASM parameter text, or None for a fixed gate; it is halved to convert
    Qiskit's convention to Qaravan's (see module docstring).
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
    """Skip a `gate <name>(...) ... { ... }` declaration by brace depth, without interpreting
    its body -- declared gates are dispatched by invocation name alone."""
    depth = first_line.count("{") - first_line.count("}")
    while depth > 0:
        next_line = next(lines)
        depth += next_line.count("{") - next_line.count("}")


def from_unitary_qasm(qasm_text: str, num_sites: int) -> Circuit:
    """Parse unitary-only QASM text into a Circuit.

    Recognizes header/qreg/barrier lines, an optional `// global_phase = <value>` comment
    (absent means phase 0), and `gate` declaration blocks. Raises NotImplementedError by name
    for any gate outside the supported vocabulary.
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


# === Export ===

_QELIB1 = frozenset(
    {
        "id",
        "h",
        "x",
        "y",
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "cx",
        "cz",
        "swap",
        "rx",
        "ry",
        "rz",
    }
)
"""Names qelib1.inc already defines. Anything else needs a `gate` declaration."""


def _build_qasm_names() -> dict[type, tuple[str, bool]]:
    """Gate class -> (qasm name, takes an angle), inverting the four import tables."""
    names = {}
    for table in (_ONE_QUBIT_FIXED, _TWO_QUBIT_FIXED):
        for qasm_name, gate_cls in table.items():
            names[gate_cls] = (qasm_name, False)
    for table in (_ONE_QUBIT_PARAMETERISED, _TWO_QUBIT_PARAMETERISED):
        for qasm_name, gate_cls in table.items():
            names[gate_cls] = (qasm_name, True)
    return names


_QASM_NAMES = _build_qasm_names()


def _qasm_line(gate) -> str:
    """One Gate -> one QASM instruction.

    Angles are doubled to reach Qiskit's convention and written as repr(float(...)), the shortest
    string that reads back as the same float64. The float() is load-bearing: a numpy scalar reprs
    as "np.float64(0.5)", which is not a QASM angle expression.
    """
    entry = _QASM_NAMES.get(type(gate))
    if entry is None:
        raise NotImplementedError(
            f"Gate '{gate.name}' has no QASM equivalent -- not part of the gate vocabulary "
            "this exporter supports."
        )
    qasm_name, parameterised = entry
    qubits = ", ".join(f"q[{index}]" for index in gate.indices)
    if not parameterised:
        return f"{qasm_name} {qubits};"
    return f"{qasm_name}({float(2 * gate.params[0])!r}) {qubits};"


def _declarations(gates: list) -> list[str]:
    """Bare `gate` declarations for every non-qelib1 name used, in first-use order.

    The body is empty: a reader dispatches on the invocation name, exactly as the importer does.
    """
    lines = []
    declared = set()
    for gate in gates:
        entry = _QASM_NAMES.get(type(gate))
        if entry is None:
            continue
        qasm_name, parameterised = entry
        if qasm_name in _QELIB1 or qasm_name in declared:
            continue
        declared.add(qasm_name)
        operands = ", ".join("abc"[: len(gate.indices)])
        signature = f"{qasm_name}(theta)" if parameterised else qasm_name
        lines.append(f"gate {signature} {operands} {{ }}")
    return lines


def to_unitary_qasm(circuit: Circuit) -> str:
    """Serialise a Circuit as unitary-only QASM text (see module docstring for the conventions).

    A GlobalPhase gate becomes the `// global_phase` comment, so a re-imported circuit carries
    it at index 0 rather than wherever it sat before -- the same unitary, a different position.
    Raises NotImplementedError for qutrit circuits and for gates outside the vocabulary.
    """
    if circuit.local_dim != 2:
        raise NotImplementedError(
            f"QASM is qubit-only; circuit has local_dim={circuit.local_dim}."
        )

    global_phase = 0.0
    gates = []
    for gate in circuit.gates:
        if gate.name == "GlobalPhase":
            global_phase += float(np.angle(gate.matrix[0, 0]))
        else:
            gates.append(gate)

    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    if abs(global_phase) > 0:
        lines.append(f"// global_phase = {global_phase!r}")
    lines.extend(_declarations(gates))
    lines.append(f"qreg q[{circuit.num_sites}];")
    for gate in gates:
        lines.append(_qasm_line(gate))
    return "\n".join(lines) + "\n"
