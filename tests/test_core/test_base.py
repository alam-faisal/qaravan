"""Smoke tests for core/base.py — structural, not physics."""
import numpy as np
import pytest

from qaravan.core.base import (
    Gate, Circuit, State, Simulator, Observable, NoiseModel,
    IncompatibleNoiseError, IncompatibleStateError,
)

# ---------------------------------------------------------------------------
# Helpers — minimal concrete subclasses for abstract class tests
# ---------------------------------------------------------------------------

class MinimalState(State):
    def expectation(self, observable):
        return 0.0
    def sample(self, shots):
        return np.array([])
    def sample_and_collapse(self, sites):
        return ("0" * len(sites), self)
    def overlap(self, other):
        return 1.0


class MinimalObservable(Observable):
    pass


class MinimalNoiseModel(NoiseModel):
    def get_kraus(self, gate):
        d = gate.matrix.shape[0]
        return [np.eye(d)]


class MinimalSimulator(Simulator):
    def _validate_state(self, state):
        if not isinstance(state, MinimalState):
            raise IncompatibleStateError

    def translate_gate(self, gate):
        return gate.matrix

    def _apply_translated_gate(self, state, translated_gate):
        pass  # no-op for structural tests


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

H_MATRIX = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X_MATRIX = np.array([[0, 1], [1, 0]], dtype=complex)

def test_gate_int_index_normalised():
    g = Gate("X", 2, X_MATRIX)
    assert g.indices == [2]

def test_gate_list_index_preserved():
    g = Gate("CX", [0, 1], np.eye(4))
    assert g.indices == [0, 1]

def test_gate_num_sites():
    assert Gate("H", 0, H_MATRIX).num_sites == 1
    assert Gate("CX", [0, 1], np.eye(4)).num_sites == 2

def test_gate_local_dim():
    assert Gate("H", 0, H_MATRIX).local_dim == 2
    assert Gate("CX", [0, 1], np.eye(4)).local_dim == 2

def test_gate_dagger_hermitian():
    h = Gate("H", 0, H_MATRIX)
    hd = h.dagger()
    assert np.allclose(hd.matrix, H_MATRIX.conj().T)
    assert "†" in hd.name

def test_gate_dagger_self_inverse():
    h = Gate("H", 0, H_MATRIX)
    hd = h.dagger()
    assert np.allclose(hd.matrix, H_MATRIX)  # H is self-inverse

def test_gate_time_default_none():
    g = Gate("X", 0, X_MATRIX)
    assert g.time is None

def test_gate_time_set():
    g = Gate("X", 0, X_MATRIX, time=0.04)
    assert g.time == pytest.approx(0.04)

def test_gate_str():
    g = Gate("X", 0, X_MATRIX)
    s = str(g)
    assert "X" in s

# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------

def _two_qubit_circuit():
    h = Gate("H", 0, H_MATRIX)
    cx = Gate("CX", [0, 1], np.eye(4))
    return Circuit([h, cx])

def test_circuit_n_inferred():
    circ = _two_qubit_circuit()
    assert circ.n == 2

def test_circuit_n_explicit():
    h = Gate("H", 0, H_MATRIX)
    circ = Circuit([h], n=3)
    assert circ.n == 3

def test_circuit_len():
    assert len(_two_qubit_circuit()) == 2

def test_circuit_getitem_gate():
    circ = _two_qubit_circuit()
    assert isinstance(circ[0], Gate)

def test_circuit_getitem_slice():
    circ = _two_qubit_circuit()
    sub = circ[0:1]
    assert isinstance(sub, Circuit)
    assert len(sub) == 1

def test_circuit_construct_layers():
    circ = _two_qubit_circuit()
    circ.construct_layers()
    assert circ.layers is not None
    # H on qubit 0 and CX on [0,1] conflict — must be in separate layers
    assert len(circ.layers) == 2

def test_circuit_layers_parallel():
    h0 = Gate("H", 0, H_MATRIX)
    h1 = Gate("H", 1, H_MATRIX)
    circ = Circuit([h0, h1])
    circ.construct_layers()
    # independent gates on different qubits should pack into one layer
    assert len(circ.layers) == 1

def test_circuit_add():
    circ = _two_qubit_circuit()
    doubled = circ + circ
    assert len(doubled) == 4

def test_circuit_mul():
    circ = _two_qubit_circuit()
    tripled = circ * 3
    assert len(tripled) == 6

def test_circuit_rmul():
    circ = _two_qubit_circuit()
    tripled = 3 * circ
    assert len(tripled) == 6

def test_circuit_dagger_reversed():
    h = Gate("H", 0, H_MATRIX)
    cx = Gate("CX", [0, 1], np.eye(4))
    circ = Circuit([h, cx])
    dag = circ.dagger()
    assert dag.gates[0].name == circ.gates[-1].dagger().name
    assert dag.gates[1].name == circ.gates[0].dagger().name

def test_circuit_dagger_gate_matrices():
    circ = _two_qubit_circuit()
    dag = circ.dagger()
    # first gate of dagger should be CX† = CX (since CX is real and unitary self-inverse)
    assert np.allclose(dag.gates[0].matrix, np.eye(4))

def test_circuit_copy_independent():
    circ = _two_qubit_circuit()
    copy = circ.copy()
    copy.gates.append(Gate("X", 0, X_MATRIX))
    assert len(circ) == 2

# ---------------------------------------------------------------------------
# State (abstract)
# ---------------------------------------------------------------------------

def test_state_cannot_instantiate():
    with pytest.raises(TypeError):
        State()

def test_state_concrete_subclass_works():
    s = MinimalState()
    assert s.expectation(None) == 0.0

# ---------------------------------------------------------------------------
# Observable (abstract)
# ---------------------------------------------------------------------------

def test_observable_indices_normalised():
    obs = MinimalObservable("Z", 0)
    assert obs.indices == [0]

def test_observable_list_indices():
    obs = MinimalObservable("ZZ", [0, 1])
    assert obs.indices == [0, 1]

# ---------------------------------------------------------------------------
# NoiseModel
# ---------------------------------------------------------------------------

def test_noisemodel_get_kraus_not_implemented_by_default():
    class BareNoise(NoiseModel):
        pass
    nm = BareNoise()
    with pytest.raises(NotImplementedError):
        nm.get_kraus(Gate("X", 0, X_MATRIX))

def test_noisemodel_get_superop_uses_kraus():
    nm = MinimalNoiseModel()
    g = Gate("X", 0, X_MATRIX)
    superop = nm.get_superop(g)
    # identity Kraus => identity superoperator
    assert np.allclose(superop, np.eye(4))

def test_noisemodel_sample_error_not_implemented():
    nm = MinimalNoiseModel()
    with pytest.raises(NotImplementedError):
        nm.sample_error(Gate("X", 0, X_MATRIX))

# ---------------------------------------------------------------------------
# Simulator (abstract)
# ---------------------------------------------------------------------------

def test_simulator_cannot_instantiate():
    with pytest.raises(TypeError):
        Simulator(Circuit([]), MinimalState())

def test_simulator_run_noiseless():
    circ = _two_qubit_circuit()
    state = MinimalState()
    sim = MinimalSimulator(circ, state)
    result = sim.run()
    assert isinstance(result, State)

def test_simulator_run_raises_on_noise():
    """Default _insert_noise must raise IncompatibleNoiseError when nm is not None."""
    circ = _two_qubit_circuit()
    state = MinimalState()
    nm = MinimalNoiseModel()
    sim = MinimalSimulator(circ, state, noise_model=nm)
    with pytest.raises(IncompatibleNoiseError):
        sim.run()

def test_simulator_run_does_not_mutate_circuit():
    circ = _two_qubit_circuit()
    n_gates_before = len(circ)
    state = MinimalState()
    MinimalSimulator(circ, state).run()
    assert len(circ) == n_gates_before
    assert circ.layers is None  # construct_layers called on copy, not original

def test_simulator_incompatible_state_raises():
    class OtherState(State):
        def expectation(self, o): return 0.0
        def sample(self, s): return np.array([])
        def sample_and_collapse(self, s): return ("0", self)
        def overlap(self, o): return 0.0

    circ = _two_qubit_circuit()
    with pytest.raises(IncompatibleStateError):
        MinimalSimulator(circ, OtherState())

def test_simulator_decompose_kwarg_accepted():
    circ = _two_qubit_circuit()
    state = MinimalState()
    sim = MinimalSimulator(circ, state, decompose=False)
    result = sim.run()
    assert isinstance(result, State)
