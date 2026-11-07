from collections.abc import Sequence
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.container import BarContainer
from matplotlib.lines import Line2D
from numpy.typing import ArrayLike, NDArray
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from lle_solver.simulations.evolution import Intensity, IntensityEvolution


def cell_edges(values: ArrayLike, *, logarithmic: bool = False) -> NDArray[np.float64]:
    """Границы ячеек на полшага от узлов в выбранном масштабе оси."""
    values = np.asarray(values, dtype=float)
    if values.size < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("Нужны хотя бы два строго возрастающих узла")
    if logarithmic and np.any(values <= 0):
        raise ValueError("Логарифмические узлы должны быть положительными")
    coordinates = np.log(values) if logarithmic else values
    midpoints = (coordinates[:-1] + coordinates[1:]) / 2
    edges = np.r_[2 * coordinates[0] - midpoints[0], midpoints,
                  2 * coordinates[-1] - midpoints[-1]]
    return np.exp(edges) if logarithmic else edges


def plot_outcome_map(
    ax: Axes, widths: ArrayLike, amplitudes: ArrayLike, matrix: ArrayLike, colors: Sequence[str],
) -> tuple[QuadMesh, PathCollection]:
    x_edges = cell_edges(widths, logarithmic=True)
    y_edges = cell_edges(amplitudes)
    image = ax.pcolormesh(
        x_edges, y_edges, matrix, cmap=ListedColormap(colors),
        norm=BoundaryNorm(np.arange(len(colors)+1)-.5, len(colors)),
    )
    x, y = np.meshgrid(widths, amplitudes)
    nodes = ax.scatter(x.ravel(), y.ravel())
    return image, nodes


def plot_intensity(ax: Axes, intensity: IntensityEvolution) -> QuadMesh:
    return ax.pcolormesh(intensity.theta, intensity.times, intensity.values, shading="nearest", rasterized=True)


def plot_profiles(ax: Axes, profiles: Sequence[Intensity]) -> list[Line2D]:
    return [line for profile in profiles for line in ax.plot(profile.theta, profile.values)]


def plot_stacked_counts(ax: Axes, coordinates: ArrayLike, counts: ArrayLike) -> list[BarContainer]:
    bottom = np.zeros(len(np.asarray(coordinates)))
    bars = []
    for values in np.asarray(counts):
        bars.append(ax.bar(coordinates, values, bottom=bottom))
        bottom += values
    return bars


def plot_separation(ax: Axes, times: ArrayLike, separation: ArrayLike) -> list[Line2D]:
    return ax.plot(times, separation)
