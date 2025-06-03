from qaravan.core import Circuit, Gate
from typing import Tuple, List
import stim
import itertools

def cnot_conversion(gate: Gate) -> Tuple[str, List[int]]:
    """ convert CNOT gate to Stim format """
    name = "CNOT"
    if gate.bottom_heavy: 
        indices = gate.indices
    else:
        indices = gate.indices[::-1]
    return name, indices

def qaravan_to_stim(circ: Circuit) -> stim.Circuit:
    """ convert a Qaravan circuit to a Stim circuit """
    stim_circ = stim.Circuit()
    for gate in circ:
        try: 
            if gate.name == "CNOT2": 
                stim_circ.append(*cnot_conversion(gate)) 
            else: 
                stim_circ.append(gate.name, gate.indices)
        except Exception:
            raise NotImplementedError(f"Gate {gate} not implemented in Stim conversion.")
    
    return stim_circ

def group_from_generators(generators: List[stim.PauliString]) -> List[stim.PauliString]:
    """ create a group from a list of generators """
    stabilizer_group = []
    for bits in itertools.product([0, 1], repeat=len(generators)):
        p = stim.PauliString(len(generators[0])) 
        for i, bit in enumerate(bits):
            if bit:
                p *= generators[i]
        stabilizer_group.append(p)

    return stabilizer_group

def is_stabilized_by(circuit: stim.Circuit, pauli: stim.PauliString) -> bool:
    """ check if a circuit stabilizes a Pauli string """
    sim = stim.TableauSimulator()
    sim.do_circuit(circuit)
    return sim.peek_observable_expectation(pauli) == 1

def string_to_stim(pstr: str) -> stim.PauliString:
    """ convert a string representation of a Pauli string to a stim.PauliString """
    return stim.PauliString(pstr.replace('x', 'X').replace('y', 'Y').replace('z', 'Z').replace('i', 'I'))

def stim_to_string(pauli: stim.PauliString) -> str:
    """ convert a stim.PauliString to a string representation """
    return str(pauli)[1:].replace('X', 'x').replace('Y', 'y').replace('Z', 'z').replace('_', 'i')