import numpy as np
from numpy.typing import ArrayLike, NDArray


def rectangular_grid(
    x: ArrayLike, y: ArrayLike, values: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Упорядоченные оси и матрица значений в узлах прямоугольной сетки."""
    x, y, values = (np.asarray(a, dtype=float) for a in (x, y, values))
    if x.ndim != 1 or x.shape != y.shape or x.shape != values.shape or not x.size:
        raise ValueError("Нужны непустые одномерные массивы одинаковой длины")
    if not all(np.isfinite(a).all() for a in (x, y, values)):
        raise ValueError("Координаты и значения должны быть конечными")
    xs, xi = np.unique(x, return_inverse=True)
    ys, yi = np.unique(y, return_inverse=True)
    positions = yi * xs.size + xi
    if np.unique(positions).size != positions.size:
        raise ValueError("Узел сетки повторяется")
    if positions.size != xs.size * ys.size:
        raise ValueError("Сетка неполная")
    matrix = np.empty((ys.size, xs.size))
    matrix[yi, xi] = values
    return xs, ys, matrix
