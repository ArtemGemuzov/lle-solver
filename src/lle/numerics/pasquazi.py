from functools import lru_cache

import numpy as np
from scipy.linalg import expm

from lle.numerics.grid import PeriodicGrid
from lle.numerics.split_schemes import StrangSplitStep
from lle.numerics.ssfm import recording_steps
from lle.physics.fields import FieldEvolution
from lle.physics.pasquazi import PasquaziEvolution, PasquaziParameters


def linear_symbol(p: PasquaziParameters, k: np.ndarray) -> np.ndarray:
    """Матрицы фурье-символа в базисе (a, b_-N, ..., b_N)."""
    q = np.arange(-p.n, p.n + 1)
    symbol = np.empty((k.size, q.size + 1, q.size + 1), dtype=np.complex128)
    symbol[:, 0, 0] = -p.kappa + 0.5j * p.zeta_a * k**2
    symbol[:, 0, 1:] = np.sqrt(p.kappa)
    symbol[:, 1:, 0] = np.sqrt(p.kappa)
    # В каждое уравнение петли входит отрицательная сумма всех супермод.
    symbol[:, 1:, 1:] = -1
    index = np.arange(1, q.size + 1)
    symbol[:, index, index] += (
        p.gain
        + 2j * np.pi * (p.detuning - q)
        - 1j * p.fsr_mismatch * k[:, None]
        + (0.5j * p.zeta_b - p.sigma) * k[:, None] ** 2
    )
    return symbol


def solve(
    p: PasquaziParameters,
    grid: PeriodicGrid,
    initial: np.ndarray,
    *,
    time_step: float,
    time_end: float,
    recording_interval: float,
) -> PasquaziEvolution:
    """Решает систему на кольце единичного периода. Строки initial: a, затем b_q."""
    steps, interval = recording_steps(time_end, recording_interval, time_step)
    if grid.length != 1 or grid.num_points < 2:
        raise ValueError("Expected a unit-period grid with at least two points")
    state = np.array(initial, dtype=np.complex128, copy=True)
    if state.shape != (2 * p.n + 2, grid.num_points) or not np.isfinite(state).all():
        raise ValueError(
            "Expected finite initial fields indexed by component and space"
        )
    symbol = linear_symbol(p, grid.wavenumbers)

    @lru_cache(maxsize=2)
    def propagator(dt: float) -> np.ndarray:
        return expm(dt * symbol)

    def linear_step(state: np.ndarray, step_size: float) -> np.ndarray:
        spectrum = np.fft.fft(state)
        evolved = np.einsum("kij,jk->ik", propagator(step_size), spectrum)
        return np.fft.ifft(evolved)

    def kerr_step(state: np.ndarray, step_size: float) -> np.ndarray:
        state[0] *= np.exp(-1j * step_size * np.abs(state[0]) ** 2)
        return state

    scheme = StrangSplitStep(linear_step, kerr_step)
    times = np.arange(0, steps + 1, interval) * time_step
    frames = np.empty((times.size, *state.shape), dtype=np.complex128)
    frames[0] = state
    for i in range(1, times.size):
        state = scheme.advance(state, interval, time_step)
        if not np.isfinite(state).all():
            raise FloatingPointError(f"Non-finite field at step {i * interval}")
        frames[i] = state
    return PasquaziEvolution(
        FieldEvolution(grid.theta, times, frames[:, 0]), frames[:, 1:]
    )
