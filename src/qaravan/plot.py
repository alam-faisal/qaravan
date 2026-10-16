"""Matplotlib styles and Gatefold-backed circuit plotting."""

import matplotlib.pyplot as plt
from gatefold import Item, Layer, Palette, StyleSpec
from gatefold import plot_circuit as plot_gatefold_circuit
from matplotlib.axes import Axes
from qaravan.core.circuit import Circuit
from qaravan.core.gates import ParametricGate

QARAVAN_PALETTE = Palette(
    styles={
        "default": StyleSpec(fill="#4C6EF5"),
        "single": StyleSpec(fill="#5F3DC4"),
        "multi": StyleSpec(fill="#087F5B"),
    }
)


def set_style(
    axes_labelsize: float = 10.5,
    tick_labelsize: int = 9,
    legend_fontsize: int = 8,
    axes_titlesize: int = 11,
    linewidth: float = 1.8,
    savefig_dpi: int = 300,
    figure_dpi: int = 300,
    grid: bool = True,
    spines_top: bool = True,
    spines_right: bool = False,
    xtick_top: bool = False,
    ytick_right: bool = False,
    spines_left: bool = True,
    ytick_left: bool = True,
) -> None:
    """Apply Qaravan's publication plot style without requiring LaTeX."""
    import scienceplots  # noqa: F401  — registers 'science' style on import

    plt.style.use(["science"])
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman", "serif"],
            "mathtext.fontset": "dejavuserif",
            "image.cmap": "plasma",
            "axes.labelsize": axes_labelsize,
            "axes.titlesize": axes_titlesize,
            "axes.grid": grid,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.spines.top": spines_top,
            "axes.spines.right": spines_right,
            "axes.spines.left": spines_left,
            "grid.color": "0.85",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.labelsize": tick_labelsize,
            "ytick.labelsize": tick_labelsize,
            "xtick.top": xtick_top,
            "ytick.right": ytick_right,
            "ytick.left": ytick_left,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": legend_fontsize,
            "legend.frameon": False,
            "lines.linewidth": linewidth,
            "lines.solid_capstyle": "round",
            "savefig.dpi": savefig_dpi,
            "figure.dpi": figure_dpi,
        }
    )


def _gate_label(gate: ParametricGate) -> str:
    params = ", ".join(f"{param:.2g}" for param in gate.params)
    return rf"${gate.name}({params})$"


def circuit_to_layers(circuit: Circuit) -> list[Layer]:
    """Convert a circuit to Gatefold layers without changing its cached layers."""
    layered_circuit = circuit.copy()
    layered_circuit.construct_layers()

    layers = []
    for gate_layer in layered_circuit.layers:
        items = []
        for gate in gate_layer:
            label = _gate_label(gate) if isinstance(gate, ParametricGate) else gate.name
            style = "single" if gate.num_sites == 1 else "multi"
            items.append(Item(tuple(str(site) for site in gate.indices), label, style))
        layers.append(Layer(items))
    return layers


def plot_circuit(
    circuit: Circuit,
    *,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    palette: Palette = QARAVAN_PALETTE,
) -> Axes:
    """Draw a circuit and return its Matplotlib axes."""
    qubit_labels = [str(site) for site in range(circuit.num_sites)]
    return plot_gatefold_circuit(
        circuit_to_layers(circuit),
        qubit_labels=qubit_labels,
        palette=palette,
        ax=ax,
        figsize=figsize,
    )
