"""Tests for Gatefold-backed Qaravan circuit plotting."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from qaravan.core.circuit import Circuit
from qaravan.core.gates import CNOT, RX, H, X, Z
from qaravan.plot import QARAVAN_PALETTE, circuit_to_layers, plot_circuit


def test_circuit_to_layers_maps_wires_labels_and_styles():
    circuit = Circuit([H(0), RX(1, 0.25), CNOT([0, 1])], num_sites=2)

    layers = circuit_to_layers(circuit)
    items = [item for layer in layers for item in layer.items]

    assert [item.qubits for item in items] == [("0",), ("1",), ("0", "1")]
    assert [item.label for item in items] == ["H", r"$RX(0.25)$", "CNOT"]
    assert [item.style for item in items] == ["single", "single", "multi"]


def test_circuit_to_layers_preserves_shared_wire_execution_order():
    circuit = Circuit([H(0), CNOT([2, 0]), X(0), Z(2)], num_sites=3)

    layers = circuit_to_layers(circuit)
    ordered = [item.label for layer in layers for item in layer.items]

    assert ordered == ["H", "CNOT", "X", "Z"]


def test_circuit_to_layers_keeps_disjoint_gates_together_in_input_order():
    circuit = Circuit([H(2), X(0), Z(4)], num_sites=5)

    layers = circuit_to_layers(circuit)

    assert len(layers) == 1
    assert [item.label for item in layers[0].items] == ["H", "X", "Z"]


def test_circuit_to_layers_does_not_mutate_cached_layers():
    circuit = Circuit([H(0), CNOT([0, 1])], num_sites=2)
    circuit.construct_layers()
    cached_layers = circuit.layers

    circuit_to_layers(circuit)

    assert circuit.layers is cached_layers


def test_plot_circuit_returns_axes_and_includes_idle_wires():
    circuit = Circuit([H(0), CNOT([0, 1])], num_sites=3)

    ax = plot_circuit(circuit)

    assert isinstance(ax, Axes)
    assert {text.get_text() for text in ax.texts} >= {"0", "1", "2", "H", "CNOT"}
    plt.close(ax.figure)


def test_plot_circuit_draws_an_empty_declared_register():
    ax = plot_circuit(Circuit([], num_sites=3))

    assert isinstance(ax, Axes)
    assert [text.get_text() for text in ax.texts] == ["0", "1", "2"]
    plt.close(ax.figure)


def test_qaravan_palette_defines_every_emitted_style():
    assert QARAVAN_PALETTE.get("default")
    assert QARAVAN_PALETTE.get("single")
    assert QARAVAN_PALETTE.get("multi")
