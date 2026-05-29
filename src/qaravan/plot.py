"""Matplotlib style configuration for Qaravan notebooks and scripts."""

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  — registers 'science' style on import


def set_style(
    axes_labelsize: int = 8,
    tick_labelsize: int = 8,
    legend_fontsize: int = 8,
    axes_titlesize: int = 10,
    linewidth: float = 1.5,
    savefig_dpi: int = 300,
    figure_dpi: int = 300,
    grid: bool = True,
    spines_top: bool = True,
    spines_right: bool = True,
    xtick_top: bool = False,
    ytick_right: bool = False,
    spines_left: bool = True,
    ytick_left: bool = False,
) -> None:
    """Apply the Qaravan matplotlib style (scienceplots 'science' + grid, no LaTeX)."""
    plt.style.use(["science", "grid"])
    plt.rcParams["text.usetex"] = False
    plt.rcParams["image.cmap"] = "cividis"

    plt.rcParams["axes.labelsize"] = axes_labelsize
    plt.rcParams["xtick.labelsize"] = tick_labelsize
    plt.rcParams["ytick.labelsize"] = tick_labelsize
    plt.rcParams["legend.fontsize"] = legend_fontsize
    plt.rcParams["axes.titlesize"] = axes_titlesize

    plt.rcParams["lines.linewidth"] = linewidth
    plt.rcParams["savefig.dpi"] = savefig_dpi
    plt.rcParams["figure.dpi"] = figure_dpi

    plt.rcParams["axes.grid"] = grid
    plt.rcParams["axes.spines.top"] = spines_top
    plt.rcParams["axes.spines.right"] = spines_right
    plt.rcParams["xtick.top"] = xtick_top
    plt.rcParams["ytick.right"] = ytick_right
    plt.rcParams["axes.spines.left"] = spines_left
    plt.rcParams["ytick.left"] = ytick_left
