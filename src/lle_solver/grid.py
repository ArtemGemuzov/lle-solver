from typing import Self

import numpy as np
from numpy.typing import NDArray


def uniform_spacing(theta: NDArray[np.float64]) -> float:
    """Возвращает шаг равномерной координатной сетки."""
    steps = np.diff(theta)
    if (steps.size == 0 or not np.isfinite(steps[0]) or steps[0] == 0
            or not np.allclose(steps, steps[0], rtol=1e-10, atol=0)):
        raise ValueError("Ожидается равномерная сетка с ненулевым шагом")
    return float(steps[0])


class PeriodicGrid:
    def __init__(self, num_points: int, length: float = 2*np.pi) -> None:
        self.num_points = num_points
        self.length = length
        self.spacing = length / num_points
        
        self.theta = np.linspace(-length / 2, length / 2, num_points, endpoint=False)
        self.wavenumbers = (
            2 * np.pi / length * np.fft.fftfreq(num_points, d=1.0 / num_points)
        )

    @classmethod
    def from_theta(cls, theta: NDArray[np.float64]) -> Self:
        """Создаёт сетку из возрастающих равномерных координат без повторной границы.

        Период равен числу узлов, умноженному на шаг. Координаты копируются
        с сохранением исходного начала отсчёта.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1 or not np.isfinite(theta).all():
            raise ValueError("Координаты должны быть одномерным массивом конечных чисел")
        spacing = uniform_spacing(theta)
        if spacing <= 0:
            raise ValueError("Координаты должны строго возрастать")
        length = theta.size * spacing
        if not np.isfinite(length):
            raise ValueError("Длина периода должна быть конечной")
        grid = cls(theta.size, length)
        grid.theta = theta.copy()
        return grid

    def distance(self, first: float, second: float) -> float:
        """Вычисляет кратчайшее расстояние между двумя координатами по кольцу."""
        arc = abs(second - first) % self.length
        return min(arc, self.length - arc)
