"""Оформление графиков и таблиц в исследовательских блокнотах."""

from pathlib import Path
from collections.abc import Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


def configure_plots() -> None:
    plt.rcParams.update({
        "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
        "figure.figsize": (6.4, 4.5), "figure.dpi": 120, "savefig.dpi": 300,
        "axes.grid": False, "lines.linewidth": 1.7, "svg.fonttype": "none",
        "axes.xmargin": .05, "axes.ymargin": .05,
    })


def save_figure(results_dir: Path, name: str, figure: Figure) -> None:
    for extension in ("png", "svg"):
        figure.savefig(results_dir / f"{name}.{extension}")


def format_log_x(ax: Axes, values: Sequence[float] = (.05, .1, .2, .4, .7)) -> None:
    ax.set_xticks(values, labels=[f"{value:g}" for value in values])
    ax.xaxis.set_minor_formatter(NullFormatter())
