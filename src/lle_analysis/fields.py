from typing import overload

import numpy as np
from numpy.typing import NDArray

from lle_solver.simulations.evolution import Field, FieldEvolution, Intensity, IntensityEvolution


def time_mask(times: NDArray[np.float64], start: float, end: float) -> NDArray[np.bool_]:
    return (times >= start) & (times <= end)


def tail_mask(
    times: NDArray[np.float64], *, fraction: float,
) -> NDArray[np.bool_]:
    if not 0 < fraction <= 1:
        raise ValueError("Доля временного интервала должна быть в пределах (0, 1]")
    start = times[-1] - fraction * (times[-1]-times[0])
    return time_mask(times, start, times[-1])


def relative_l2_error(
    values: NDArray[np.complex128], reference: NDArray[np.complex128],
) -> NDArray[np.float64]:
    """Относительное отличие по последней оси; 0/0 = 0, ненулевое/0 = inf."""
    difference = np.linalg.norm(values-reference, axis=-1)
    scale = np.linalg.norm(reference, axis=-1)
    error = np.full(np.shape(difference), np.inf, dtype=float)
    np.divide(difference, scale, out=error, where=scale != 0)
    return np.where((scale == 0) & (difference == 0), 0., error)


@overload
def relative_field_l2_error(field: Field, reference: Field) -> float: ...
@overload
def relative_field_l2_error(field: FieldEvolution, reference: Field) -> NDArray[np.float64]: ...


def relative_field_l2_error(field: Field | FieldEvolution, reference: Field) -> float | NDArray[np.float64]:
    """Относительная L2-ошибка полей на совпадающих пространственных сетках."""
    if not np.array_equal(field.theta, reference.theta):
        raise ValueError("Сетки полей должны совпадать")
    if isinstance(field, Field):
        return float(relative_l2_error(field.values, reference.values))
    return relative_l2_error(field.values, reference.values)


@overload
def spatial_contrast(intensity: Intensity) -> float: ...
@overload
def spatial_contrast(intensity: IntensityEvolution) -> NDArray[np.float64]: ...


def spatial_contrast(intensity: Intensity | IntensityEvolution) -> float | NDArray[np.float64]:
    """Контраст по последней оси; нулевой профиль имеет нулевой контраст."""
    values = intensity.values
    mean = values.mean(axis=-1)
    result = np.zeros(np.shape(mean), dtype=float)
    np.divide(np.ptp(values, axis=-1), mean, out=result, where=mean != 0)
    return float(result) if isinstance(intensity, Intensity) else result
