"""Начальные поля, общие для численных методов и PINN."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class GaussianPulse:
    amplitude: float
    width: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.width) or self.width <= 0:
            raise ValueError("Pulse width must be finite and positive")

    def __call__(self, theta: NDArray[np.floating]) -> NDArray[np.complex128]:
        return (self.amplitude * np.exp(-0.5 * (theta / self.width) ** 2)).astype(
            np.complex128
        )
