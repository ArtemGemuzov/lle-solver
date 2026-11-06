from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self, overload

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates


DEFAULT_TIME_TOLERANCE = 1e-10


def _validate_axis(axis: NDArray[np.float64]) -> None:
    if axis.ndim != 1 or axis.size == 0 or axis.dtype != np.float64:
        raise ValueError("Ось должна быть непустым вектором float64")


@dataclass(frozen=True, eq=False)
class Intensity:
    """Пространственный профиль интенсивности в один момент времени."""

    theta: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        _validate_axis(self.theta)
        if self.values.shape != self.theta.shape or self.values.dtype != np.float64:
            raise ValueError("Интенсивность должна содержать одно значение float64 на узел сетки")
        if not np.all(self.values >= 0):
            raise ValueError("Интенсивность должна быть неотрицательной")


@dataclass(frozen=True, eq=False)
class Field:
    """Комплексное поле на пространственной сетке в один момент времени."""

    theta: NDArray[np.float64]
    values: NDArray[np.complex128]

    def __post_init__(self) -> None:
        _validate_axis(self.theta)
        if self.values.shape != self.theta.shape or self.values.dtype != np.complex128:
            raise ValueError("Поле должно содержать одно значение complex128 на узел сетки")

    @property
    def intensity(self) -> Intensity:
        return Intensity(self.theta, np.abs(self.values) ** 2)

    def relative_to(self, origin: float) -> Self:
        """Возвращает то же поле в координатах с началом в origin."""
        return type(self)(self.theta - origin, self.values)

    def field_at(self, theta: NDArray[np.float64], *, interpolation_order: int = 3) -> Self:
        """Возвращает поле в точках theta, интерполируя исходную равномерную сетку по кольцу."""
        _validate_axis(theta)
        if self.theta.size == 1:
            return type(self)(theta, np.full(theta.shape, self.values[0], dtype=np.complex128))
        steps = np.diff(self.theta)
        if steps[0] == 0 or not np.allclose(steps, steps[0], rtol=1e-10, atol=0):
            raise ValueError("Для периодической интерполяции нужна равномерная сетка")
        indices = (theta - self.theta[0]) / steps[0]
        values = map_coordinates(self.values, indices[None, :], order=interpolation_order, mode="grid-wrap")
        return type(self)(theta, values)


TimeSelection = slice | NDArray[np.bool_]
TimeIndex = int | np.integer


def _check_selection(key: TimeSelection, times: NDArray[np.float64]) -> None:
    if isinstance(key, slice):
        return
    if not isinstance(key, np.ndarray) or key.dtype != np.bool_ or key.shape != times.shape:
        raise IndexError("Ожидается срез или логическая маска по оси времени")


@dataclass(frozen=True, eq=False)
class IntensityEvolution:
    """История интенсивности на общей пространственной сетке."""

    theta: NDArray[np.float64]
    times: NDArray[np.float64]
    values: NDArray[np.float64]

    def __post_init__(self) -> None:
        _validate_axis(self.theta)
        _validate_axis(self.times)
        if self.values.shape != (self.times.size, self.theta.size) or self.values.dtype != np.float64:
            raise ValueError("Ожидается матрица интенсивности float64 по времени и пространству")
        if not np.all(self.values >= 0):
            raise ValueError("Интенсивность должна быть неотрицательной")

    @overload
    def __getitem__(self, key: TimeIndex) -> Intensity: ...
    @overload
    def __getitem__(self, key: TimeSelection) -> Self: ...

    def __getitem__(self, key: TimeIndex | TimeSelection) -> Intensity | Self:
        if isinstance(key, (int, np.integer)):
            if isinstance(key, bool):
                raise IndexError("Ожидается целый индекс снимка")
            return Intensity(self.theta, self.values[key])
        _check_selection(key, self.times)
        return type(self)(self.theta, self.times[key], self.values[key])

    def __iter__(self) -> Iterator[Intensity]:
        for index in range(self.times.size):
            yield self[index]

    def single(self) -> Intensity:
        """Возвращает профиль, если выбран ровно один момент времени."""
        if self.times.size != 1:
            raise ValueError(f"Ожидался один снимок, получено {self.times.size}")
        return self[0]


@dataclass(frozen=True, eq=False)
class FieldEvolution:
    theta: NDArray[np.float64]
    times: NDArray[np.float64]
    values: NDArray[np.complex128]

    def __post_init__(self) -> None:
        _validate_axis(self.times)
        _validate_axis(self.theta)
        if self.values.dtype != np.complex128:
            raise ValueError("Поле должно иметь тип complex128")
        if self.values.shape != (self.times.size, self.theta.size):
            raise ValueError("Форма поля должна совпадать с размерами осей времени и пространства")

    @overload
    def __getitem__(self, key: TimeIndex) -> Field: ...
    @overload
    def __getitem__(self, key: TimeSelection) -> Self: ...

    def __getitem__(self, key: TimeIndex | TimeSelection) -> Field | Self:
        if isinstance(key, (int, np.integer)):
            if isinstance(key, bool):
                raise IndexError("Ожидается целый индекс снимка")
            return Field(self.theta, self.values[key])
        _check_selection(key, self.times)
        return type(self)(self.theta, self.times[key], self.values[key])

    def __iter__(self) -> Iterator[Field]:
        for index in range(self.times.size):
            yield self[index]

    def single(self) -> Field:
        """Возвращает поле, если выбран ровно один момент времени."""
        if self.times.size != 1:
            raise ValueError(f"Ожидался один снимок, получено {self.times.size}")
        return self[0]

    def at_time(self, time: float, *, tolerance: float = DEFAULT_TIME_TOLERANCE) -> Field:
        """Возвращает ближайший сохранённый снимок в пределах абсолютного допуска по времени."""
        index = int(np.argmin(np.abs(self.times - time)))
        if not np.isclose(self.times[index], time, rtol=0, atol=tolerance):
            raise ValueError("Нет снимка в момент сравнения")
        return self[index]

    @property
    def intensity(self) -> IntensityEvolution:
        return IntensityEvolution(self.theta, self.times, np.abs(self.values) ** 2)
