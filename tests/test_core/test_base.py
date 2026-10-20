"""Smoke tests for core/base.py — structural, not physics."""

import numpy as np
import pytest
from qaravan.core.base import (
    Gate,
    IncompatibleNoiseError,
    IncompatibleStateError,
    NoiseModel,
    Observable,
    Simulator,
    State,
)
from qaravan.core.circuit import Circuit, _embed_gate
from qaravan.core.gates import MatrixGate

# ---------------------------------------------------------------------------
# Helpers — minimal concrete subclasses for abstract class tests
# ---------------------------------------------------------------------------


class MinimalState(State):
    @property
    def default_simulator(self):
        return MinimalSimulator

    def expectation(self, observable):
        return 0.0

    def sample(self, shots):
        return np.array([])

    def measure_and_collapse(self, sites):
        return (self, "0" * len(sites))

    def overlap(self, other):
        return 1.0

    def __repr__(self):
        return "MinimalState()"


class MinimalObservable(Observable):
    def to_matrix(self):
        return np.eye(2, dtype=complex)


class MinimalNoiseModel(NoiseModel):
    @property
    def gate_dependent(self) -> bool:
        return False

    def get_kraus(self, *args, **kwargs):
        # returns identity Kraus for any call signature
        return [np.eye(2)]


class MinimalSimulator(Simulator):
    def _validate_state(self, state):
        if not isinstance(state, MinimalState):
            raise IncompatibleStateError

    def _validate_noise(self, noise_model):
        raise IncompatibleNoiseError("MinimalSimulator is noiseless")

    def _insert_noise(self, circuit, noise_model):
        return circuit

    def translate_gate(self, gate):
        return gate.matrix

    def _apply_translated_gate(self, state, translated_gate):
        pass  # no-op for structural tests


# ---------------------------------------------------------------------------
# Gate / MatrixGate
# ---------------------------------------------------------------------------

H_MATRIX = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
X_MATRIX = np.array([[0, 1], [1, 0]], dtype=complex)


def test_gate_int_index_normalised():
    g = MatrixGate("X", 2, X_MATRIX)
    assert g.indices == [2]


def test_gate_list_index_preserved():
    g = MatrixGate("CX", [0, 1], np.eye(4))
    assert g.indices == [0, 1]


def test_gate_num_sites():
    assert MatrixGate("H", 0, H_MATRIX).num_sites == 1
    assert MatrixGate("CX", [0, 1], np.eye(4)).num_sites == 2


def test_gate_local_dim():
    assert MatrixGate("H", 0, H_MATRIX).local_dim == 2
    assert MatrixGate("CX", [0, 1], np.eye(4)).local_dim == 2


def test_gate_dagger_hermitian():
    h = MatrixGate("H", 0, H_MATRIX)
    hd = h.dagger()
    assert np.allclose(hd.matrix, H_MATRIX.conj().T)
    assert "†" in hd.name


def test_gate_dagger_self_inverse():
    h = MatrixGate("H", 0, H_MATRIX)
    hd = h.dagger()
    assert np.allclose(hd.matrix, H_MATRIX)  # H is self-inverse


def test_gate_time_default_none():
    g = MatrixGate("X", 0, X_MATRIX)
    assert g.time is None


def test_gate_time_set():
    g = MatrixGate("X", 0, X_MATRIX, time=0.04)
    assert g.time == pytest.approx(0.04)


def test_gate_str():
    g = MatrixGate("X", 0, X_MATRIX)
    s = str(g)
    assert "X" in s


def test_gate_is_abstract():
    with pytest.raises(TypeError):
        Gate("X", 0)  # type: ignore[abstract]


def test_matrix_gate_isinstance_gate():
    g = MatrixGate("H", 0, H_MATRIX)
    assert isinstance(g, Gate)


# ---------------------------------------------------------------------------
# Circuit
# ---------------------------------------------------------------------------


def _two_qubit_circuit():
    h = MatrixGate("H", 0, H_MATRIX)
    cx = MatrixGate("CX", [0, 1], np.eye(4))
    return Circuit([h, cx])


def test_circuit_num_sites_inferred():
    circ = _two_qubit_circuit()
    assert circ.num_sites == 2


def test_circuit_num_sites_explicit():
    h = MatrixGate("H", 0, H_MATRIX)
    circ = Circuit([h], num_sites=3)
    assert circ.num_sites == 3


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
    h0 = MatrixGate("H", 0, H_MATRIX)
    h1 = MatrixGate("H", 1, H_MATRIX)
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
    h = MatrixGate("H", 0, H_MATRIX)
    cx = MatrixGate("CX", [0, 1], np.eye(4))
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
    copy.gates.append(MatrixGate("X", 0, X_MATRIX))
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


def test_state_apply():
    circ = _two_qubit_circuit()
    state = MinimalState()
    result = state.apply(circ)
    assert isinstance(result, State)


def test_state_partial_overlap_not_implemented_by_default():
    """Base-class partial_overlap raises NotImplementedError; same pattern as NoiseModel.get_kraus."""
    state = MinimalState()
    with pytest.raises(NotImplementedError):
        state.partial_overlap(state, skip=[])


def test_state_overlap_derived_from_partial_overlap():
    """A State that implements only partial_overlap gets overlap() for free."""

    class PartialState(State):
        """Minimal State that implements partial_overlap but not overlap directly."""

        @property
        def default_simulator(self):
            return MinimalSimulator

        def expectation(self, o):
            return 0.0

        def sample(self, s):
            return np.array([])

        def measure_and_collapse(self, s):
            return (self, "0" * len(s))

        def partial_overlap(self, other, skip):
            # returns a (1,1) array with 1.0 for any inputs (trivial mock)
            return np.array([[1.0 + 0j]])

        def __repr__(self):
            return "PartialState()"

    s = PartialState()
    # overlap() should call partial_overlap(other, skip=[])[0,0]
    result = s.overlap(s)
    assert np.isclose(result, 1.0 + 0j)


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
        @property
        def gate_dependent(self):
            return False

    nm = BareNoise()
    with pytest.raises(NotImplementedError):
        nm.get_kraus()


def test_noisemodel_get_superop_uses_kraus():
    nm = MinimalNoiseModel()
    superop = nm.get_superop()
    # identity Kraus => identity superoperator
    assert np.allclose(superop, np.eye(4))


def test_noisemodel_sample_error_not_implemented():
    nm = MinimalNoiseModel()
    with pytest.raises(NotImplementedError):
        nm.sample_error()


def test_noisemodel_gate_dependent_abstract():
    """NoiseModel without gate_dependent cannot be instantiated."""

    class BareNoise(NoiseModel):
        pass

    with pytest.raises(TypeError):
        BareNoise()


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


def test_simulator_private_prepare_circuit_copies_and_constructs_layers():
    circuit = _two_qubit_circuit()

    prepared = MinimalSimulator(circuit, MinimalState())._prepare_circuit()

    assert prepared is not circuit
    assert prepared.gates == circuit.gates
    assert prepared.layers is not None
    assert circuit.layers is None


def test_simulator_private_evolve_copies_state_and_applies_prepared_layers():
    class RecordingState(MinimalState):
        def __init__(self):
            self.applied = []

    class RecordingSimulator(MinimalSimulator):
        def translate_gate(self, gate):
            return gate.name

        def _apply_translated_gate(self, state, translated_gate):
            state.applied.append(translated_gate)

    circuit = _two_qubit_circuit()
    simulator = RecordingSimulator(circuit, RecordingState())
    prepared = simulator._prepare_circuit()

    result = simulator._evolve(simulator.initial_state, prepared)

    assert result is not simulator.initial_state
    assert result.applied == ["H", "CX"]
    assert simulator.initial_state.applied == []


def test_simulator_validates_noise_at_construction():
    """_validate_noise runs in __init__, so rejection happens before any evolution."""
    circ = _two_qubit_circuit()
    with pytest.raises(IncompatibleNoiseError, match="MinimalSimulator"):
        MinimalSimulator(circ, MinimalState(), noise_model=MinimalNoiseModel())


def test_simulator_skips_noise_validation_without_noise():
    """noise_model=None never reaches _validate_noise, so noiseless runs always work."""
    MinimalSimulator(_two_qubit_circuit(), MinimalState()).run()


@pytest.mark.parametrize("missing", ["_validate_noise", "_insert_noise"])
def test_simulator_requires_both_noise_hooks(missing):
    """Both hooks are abstract, so accepting noise can never silently drop it."""
    methods = {
        "_validate_state": lambda self, state: None,
        "_validate_noise": lambda self, noise_model: None,
        "_insert_noise": lambda self, circuit, noise_model: circuit,
        "translate_gate": lambda self, gate: gate.matrix,
        "_apply_translated_gate": lambda self, state, translated_gate: None,
    }
    del methods[missing]
    incomplete = type("IncompleteSimulator", (Simulator,), methods)
    with pytest.raises(TypeError, match=missing):
        incomplete(_two_qubit_circuit(), MinimalState())


def test_simulator_accepts_noise_rng_without_noise():
    """noise_rng is a uniform constructor argument, harmless when no noise is sampled."""
    sim = MinimalSimulator(_two_qubit_circuit(), MinimalState(), noise_rng=3)
    assert isinstance(sim.noise_rng, np.random.Generator)
    sim.run()


def test_simulator_keeps_noise_generator_by_identity():
    """A passed Generator is reused, not re-seeded, so repeated apply() calls advance one stream."""
    noise_generator = np.random.default_rng(0)
    sim = MinimalSimulator(
        _two_qubit_circuit(), MinimalState(), noise_rng=noise_generator
    )
    assert sim.noise_rng is noise_generator


def test_simulator_noise_rng_is_keyword_only():
    """A positional noise_rng would silently land in the wrong slot, so it is rejected."""
    with pytest.raises(TypeError):
        MinimalSimulator(_two_qubit_circuit(), MinimalState(), None, False, 3)


def test_noise_model_implements_detects_overrides():
    """implements() distinguishes overridden methods from inherited NotImplementedError stubs."""
    nm = MinimalNoiseModel()
    assert nm.implements("get_kraus")
    assert not nm.implements("sample_error")
    assert not nm.implements("get_superop")  # inherited concrete default, not overridden


def test_simulator_run_does_not_mutate_circuit():
    circ = _two_qubit_circuit()
    n_gates_before = len(circ)
    state = MinimalState()
    MinimalSimulator(circ, state).run()
    assert len(circ) == n_gates_before
    assert circ.layers is None  # construct_layers called on copy, not original


def test_simulator_incompatible_state_raises():
    class OtherState(State):
        @property
        def default_simulator(self):
            return MinimalSimulator

        def expectation(self, o):
            return 0.0

        def sample(self, s):
            return np.array([])

        def measure_and_collapse(self, s):
            return (self, "0")

        def overlap(self, o):
            return 0.0

        def __repr__(self):
            return "OtherState()"

    circ = _two_qubit_circuit()
    with pytest.raises(IncompatibleStateError):
        MinimalSimulator(circ, OtherState())


def test_simulator_decompose_kwarg_accepted():
    circ = _two_qubit_circuit()
    state = MinimalState()
    sim = MinimalSimulator(circ, state, decompose=False)
    result = sim.run()
    assert isinstance(result, State)


# ---------------------------------------------------------------------------
# _embed_gate and Circuit.to_matrix
# ---------------------------------------------------------------------------

CNOT_MATRIX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)


def test_embed_gate_single_qubit_first():
    h = MatrixGate("H", 0, H_MATRIX)
    result = _embed_gate(h, num_sites=2, local_dim=2)
    expected = np.kron(H_MATRIX, np.eye(2))
    assert np.allclose(result, expected)


def test_embed_gate_single_qubit_second():
    h = MatrixGate("H", 1, H_MATRIX)
    result = _embed_gate(h, num_sites=2, local_dim=2)
    expected = np.kron(np.eye(2), H_MATRIX)
    assert np.allclose(result, expected)


def test_embed_gate_two_qubit_contiguous():
    cx = MatrixGate("CX", [0, 1], CNOT_MATRIX)
    result = _embed_gate(cx, num_sites=2, local_dim=2)
    assert np.allclose(result, CNOT_MATRIX)


def test_embed_gate_noncontiguous_cnot_02():
    """CNOT on qubits 0 and 2 of a 3-qubit system: |100⟩ → |101⟩."""
    cx = MatrixGate("CX", [0, 2], CNOT_MATRIX)
    U = _embed_gate(cx, num_sites=3, local_dim=2)

    # |100⟩ = index 4, should map to |101⟩ = index 5
    psi_in = np.zeros(8)
    psi_in[4] = 1.0
    psi_out = U @ psi_in
    assert np.allclose(psi_out[5], 1.0)
    assert np.allclose(np.linalg.norm(psi_out), 1.0)

    # |000⟩ should be unchanged
    psi_in2 = np.zeros(8)
    psi_in2[0] = 1.0
    assert np.allclose(U @ psi_in2, psi_in2)


def test_embed_gate_noncontiguous_unitary():
    """Embedded non-contiguous gate must be unitary."""
    cx = MatrixGate("CX", [0, 2], CNOT_MATRIX)
    U = _embed_gate(cx, num_sites=3, local_dim=2)
    assert np.allclose(U @ U.conj().T, np.eye(8), atol=1e-12)


def test_to_matrix_single_qubit():
    h = MatrixGate("H", 0, H_MATRIX)
    circ = Circuit([h], num_sites=1)
    assert np.allclose(circ.to_matrix(), H_MATRIX)


def test_to_matrix_h_on_qubit0_of_2():
    h = MatrixGate("H", 0, H_MATRIX)
    circ = Circuit([h], num_sites=2)
    assert np.allclose(circ.to_matrix(), np.kron(H_MATRIX, np.eye(2)))


def test_to_matrix_ghz_circuit():
    """H(0) then CNOT(0,1) on |00⟩ should give (|00⟩+|11⟩)/√2."""
    h = MatrixGate("H", 0, H_MATRIX)
    cx = MatrixGate("CX", [0, 1], CNOT_MATRIX)
    circ = Circuit([h, cx], num_sites=2)
    U = circ.to_matrix()

    psi_in = np.array([1, 0, 0, 0], dtype=complex)
    psi_out = U @ psi_in
    expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert np.allclose(psi_out, expected)
