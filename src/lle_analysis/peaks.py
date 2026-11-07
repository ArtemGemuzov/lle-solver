import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from lle_solver.grid import uniform_spacing
from lle_solver.simulations.evolution import Intensity


DEFAULT_PEAK_PROMINENCE = 0.25


def peak_indices(
    intensity: Intensity, *, prominence_ratio: float = DEFAULT_PEAK_PROMINENCE,
) -> NDArray[np.intp]:
    """Индексы пиков на кольце с выраженностью не ниже prominence_ratio * размах."""
    values = intensity.values
    span = np.ptp(values)
    n = len(values)
    indices, _ = find_peaks(np.tile(values, 3), prominence=prominence_ratio * span)
    return indices[(indices >= n) & (indices < 2*n)] - n


def peak_position(intensity: Intensity, index: int) -> float:
    """Уточняет координату вершины по параболе через узел и двух соседей на кольце."""
    values = intensity.values
    theta = intensity.theta
    if theta.size == 1:
        return float(theta[index])
    spacing = uniform_spacing(theta)
    left = values[(index - 1) % len(values)]
    right = values[(index + 1) % len(values)]
    curvature = left - 2 * values[index] + right
    if curvature == 0:
        return float(theta[index])
    shift = 0.5 * (left - right) / curvature
    return float(theta[index] + spacing * shift)
