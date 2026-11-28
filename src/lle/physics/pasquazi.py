from dataclasses import dataclass
from math import isfinite, tau

import numpy as np
from numpy.typing import NDArray

from lle.physics.fields import FieldEvolution


@dataclass(frozen=True)
class PasquaziParameters:
    """Параметры уравнений S16 и S17 в нормировке Bao и соавторов."""

    n: int = 0
    zeta_a: float = 1.25e-4
    zeta_b: float = 3.5e-4
    sigma: float = 1.5e-4
    kappa: float = tau
    gain: float = 0.14
    detuning: float = 0.4763
    fsr_mismatch: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError("Supermode truncation n must be a nonnegative integer")
        if not all(
            isfinite(v)
            for v in (
                self.zeta_a,
                self.zeta_b,
                self.sigma,
                self.kappa,
                self.gain,
                self.detuning,
                self.fsr_mismatch,
            )
        ):
            raise ValueError("Equation coefficients must be finite")
        if self.kappa < 0 or self.sigma < 0:
            raise ValueError("Coupling and spectral filtering must be nonnegative")


@dataclass(frozen=True, eq=False)
class PasquaziEvolution:
    """Поле кольца и поля петли с осями (время, q от -N до N, координата)."""

    ring: FieldEvolution
    loop: NDArray[np.complex128]

    def __post_init__(self) -> None:
        if (
            self.loop.ndim != 3
            or self.loop.shape[1] % 2 != 1
            or self.loop.shape[::2] != self.ring.values.shape
            or self.loop.dtype != np.complex128
        ):
            raise ValueError(
                "Expected complex128 loop fields indexed by time, supermode and space"
            )
