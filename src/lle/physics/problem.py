from collections.abc import Callable
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from lle.physics.equation import LLEParameters

InitialField = Callable[[NDArray[np.floating]], NDArray[np.complex128]]


@dataclass(frozen=True)
class RingDomain:
    """Кольцо длины length на интервале времени от нуля до time_end."""

    length: float
    time_end: float

    def __post_init__(self) -> None:
        if not isfinite(self.length) or self.length <= 0:
            raise ValueError("Ring length must be finite and positive")
        if not isfinite(self.time_end) or self.time_end <= 0:
            raise ValueError("End time must be finite and positive")


@dataclass(frozen=True)
class LLEProblem:
    """Постановка LLE на кольце для интервала времени от нуля до time_end.

    initial принимает вектор координат theta и возвращает комплексное поле той же формы.
    """

    equation: LLEParameters
    initial: InitialField
    length: float
    time_end: float

    @property
    def domain(self) -> RingDomain:
        return RingDomain(self.length, self.time_end)

    def __post_init__(self) -> None:
        RingDomain(self.length, self.time_end)
