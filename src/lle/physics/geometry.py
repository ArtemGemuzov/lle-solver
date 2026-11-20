import numpy as np
from numpy.typing import NDArray


def uniform_spacing(theta: NDArray[np.float64]) -> float:
    """Возвращает шаг равномерной координатной сетки."""
    steps = np.diff(theta)
    if (
        steps.size == 0
        or not np.isfinite(steps[0])
        or steps[0] == 0
        or not np.allclose(steps, steps[0], rtol=1e-10, atol=0)
    ):
        raise ValueError("Expected a uniform grid with non-zero spacing")
    return float(steps[0])
