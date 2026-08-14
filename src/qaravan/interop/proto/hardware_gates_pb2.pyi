from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ZeroQubitGate(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ZERO_QUBIT_GATE_UNSPECIFIED: _ClassVar[ZeroQubitGate]
    GLOBAL_PHASE: _ClassVar[ZeroQubitGate]

class OneQubitGateFixed(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ONE_QUBIT_GATE_FIXED_UNSPECIFIED: _ClassVar[OneQubitGateFixed]
    H: _ClassVar[OneQubitGateFixed]
    X_GATE: _ClassVar[OneQubitGateFixed]
    Y_GATE: _ClassVar[OneQubitGateFixed]
    Z_GATE: _ClassVar[OneQubitGateFixed]
    S: _ClassVar[OneQubitGateFixed]
    Sdg: _ClassVar[OneQubitGateFixed]
    T: _ClassVar[OneQubitGateFixed]
    Tdg: _ClassVar[OneQubitGateFixed]
    SX: _ClassVar[OneQubitGateFixed]
    SXdg: _ClassVar[OneQubitGateFixed]

class OneQubitGateParameterised(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ONE_QUBIT_GATE_PARAMETERISED_UNSPECIFIED: _ClassVar[OneQubitGateParameterised]
    Rx: _ClassVar[OneQubitGateParameterised]
    Ry: _ClassVar[OneQubitGateParameterised]
    Rz: _ClassVar[OneQubitGateParameterised]
    P: _ClassVar[OneQubitGateParameterised]
    U3: _ClassVar[OneQubitGateParameterised]
    PhasedXZ: _ClassVar[OneQubitGateParameterised]

class TwoQubitGateFixed(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TWO_QUBIT_GATE_FIXED_UNSPECIFIED: _ClassVar[TwoQubitGateFixed]
    CZ: _ClassVar[TwoQubitGateFixed]
    ISWAP: _ClassVar[TwoQubitGateFixed]
    CX: _ClassVar[TwoQubitGateFixed]
    SWAP: _ClassVar[TwoQubitGateFixed]
    ECR: _ClassVar[TwoQubitGateFixed]

class TwoQubitGateParameterised(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TWO_QUBIT_GATE_PARAMETERISED_UNSPECIFIED: _ClassVar[TwoQubitGateParameterised]
    MS: _ClassVar[TwoQubitGateParameterised]
    RXX: _ClassVar[TwoQubitGateParameterised]
    RYY: _ClassVar[TwoQubitGateParameterised]
    RZZ: _ClassVar[TwoQubitGateParameterised]
    XY: _ClassVar[TwoQubitGateParameterised]
ZERO_QUBIT_GATE_UNSPECIFIED: ZeroQubitGate
GLOBAL_PHASE: ZeroQubitGate
ONE_QUBIT_GATE_FIXED_UNSPECIFIED: OneQubitGateFixed
H: OneQubitGateFixed
X_GATE: OneQubitGateFixed
Y_GATE: OneQubitGateFixed
Z_GATE: OneQubitGateFixed
S: OneQubitGateFixed
Sdg: OneQubitGateFixed
T: OneQubitGateFixed
Tdg: OneQubitGateFixed
SX: OneQubitGateFixed
SXdg: OneQubitGateFixed
ONE_QUBIT_GATE_PARAMETERISED_UNSPECIFIED: OneQubitGateParameterised
Rx: OneQubitGateParameterised
Ry: OneQubitGateParameterised
Rz: OneQubitGateParameterised
P: OneQubitGateParameterised
U3: OneQubitGateParameterised
PhasedXZ: OneQubitGateParameterised
TWO_QUBIT_GATE_FIXED_UNSPECIFIED: TwoQubitGateFixed
CZ: TwoQubitGateFixed
ISWAP: TwoQubitGateFixed
CX: TwoQubitGateFixed
SWAP: TwoQubitGateFixed
ECR: TwoQubitGateFixed
TWO_QUBIT_GATE_PARAMETERISED_UNSPECIFIED: TwoQubitGateParameterised
MS: TwoQubitGateParameterised
RXX: TwoQubitGateParameterised
RYY: TwoQubitGateParameterised
RZZ: TwoQubitGateParameterised
XY: TwoQubitGateParameterised

class OneQubitGateType(_message.Message):
    __slots__ = ("fixed", "parameterised")
    FIXED_FIELD_NUMBER: _ClassVar[int]
    PARAMETERISED_FIELD_NUMBER: _ClassVar[int]
    fixed: OneQubitGateFixed
    parameterised: OneQubitGateParameterised
    def __init__(self, fixed: _Optional[_Union[OneQubitGateFixed, str]] = ..., parameterised: _Optional[_Union[OneQubitGateParameterised, str]] = ...) -> None: ...

class TwoQubitGateType(_message.Message):
    __slots__ = ("fixed", "parameterised")
    FIXED_FIELD_NUMBER: _ClassVar[int]
    PARAMETERISED_FIELD_NUMBER: _ClassVar[int]
    fixed: TwoQubitGateFixed
    parameterised: TwoQubitGateParameterised
    def __init__(self, fixed: _Optional[_Union[TwoQubitGateFixed, str]] = ..., parameterised: _Optional[_Union[TwoQubitGateParameterised, str]] = ...) -> None: ...

class Gate(_message.Message):
    __slots__ = ("zero_qubit_gate", "one_qubit_gate", "two_qubit_gate", "qubits", "angle", "angles", "phase")
    ZERO_QUBIT_GATE_FIELD_NUMBER: _ClassVar[int]
    ONE_QUBIT_GATE_FIELD_NUMBER: _ClassVar[int]
    TWO_QUBIT_GATE_FIELD_NUMBER: _ClassVar[int]
    QUBITS_FIELD_NUMBER: _ClassVar[int]
    ANGLE_FIELD_NUMBER: _ClassVar[int]
    ANGLES_FIELD_NUMBER: _ClassVar[int]
    PHASE_FIELD_NUMBER: _ClassVar[int]
    zero_qubit_gate: ZeroQubitGate
    one_qubit_gate: OneQubitGateType
    two_qubit_gate: TwoQubitGateType
    qubits: _containers.RepeatedScalarFieldContainer[str]
    angle: float
    angles: _containers.RepeatedScalarFieldContainer[float]
    phase: float
    def __init__(self, zero_qubit_gate: _Optional[_Union[ZeroQubitGate, str]] = ..., one_qubit_gate: _Optional[_Union[OneQubitGateType, _Mapping]] = ..., two_qubit_gate: _Optional[_Union[TwoQubitGateType, _Mapping]] = ..., qubits: _Optional[_Iterable[str]] = ..., angle: _Optional[float] = ..., angles: _Optional[_Iterable[float]] = ..., phase: _Optional[float] = ...) -> None: ...
