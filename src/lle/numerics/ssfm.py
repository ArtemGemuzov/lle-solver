from functools import partial
from math import isfinite

import numpy as np

from lle.numerics.grid import PeriodicGrid
from lle.numerics.integrators import rk4_step
from lle.numerics.lle import linear_operator, nonlinear_term
from lle.numerics.split_schemes import SplitScheme, StrangSplitStep
from lle.numerics.steps import FourierStep, IntegratorStep
from lle.numerics.types import State
from lle.physics.fields import FieldEvolution
from lle.physics.problem import LLEProblem


def recording_steps(time_end: float, interval: float, dt: float) -> tuple[int, int]:
    if not isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be finite and positive")
    counts = []

    for duration in (time_end, interval):
        if not isfinite(duration):
            raise ValueError("Recording times must be finite")
        count = round(duration / dt)
        if count < 1 or not np.isclose(count * dt, duration, rtol=0, atol=1e-12):
            raise ValueError("Times must be positive multiples of dt")
        counts.append(count)

    total, interval_steps = counts
    if total % interval_steps:
        raise ValueError("End time must be a multiple of the recording interval")

    return total, interval_steps


def solve_strang_rk4(
    problem: LLEProblem, *, num_points: int, time_step: float, recording_interval: float
) -> FieldEvolution:
    """Решает LLE расщеплением Стрэнга с шагом RK4 для нелинейной части."""
    if num_points < 2:
        raise ValueError("Spatial grid requires at least two points")
    dt = time_step
    total, record_step = recording_steps(problem.time_end, recording_interval, dt)
    grid = PeriodicGrid(num_points, problem.length)
    scheme = StrangSplitStep(
        FourierStep(partial(linear_operator, problem.equation), grid.wavenumbers),
        IntegratorStep(partial(nonlinear_term, problem.equation), rk4_step),
    )

    return solve(
        scheme,
        grid,
        problem.initial(grid.theta),
        num_steps=total,
        step_size=dt,
        record_every=record_step,
    )


def solve(
    scheme: SplitScheme,
    grid: PeriodicGrid,
    state: State,
    *,
    num_steps: int,
    step_size: float,
    record_every: int,
) -> FieldEvolution:
    """Записывает поле на шагах от нуля до num_steps с интервалом record_every.

    Последний шаг включается и при неполном интервале. Время отсчитывается
    от переданного состояния.
    """
    if (
        isinstance(num_steps, bool)
        or not isinstance(num_steps, (int, np.integer))
        or num_steps < 0
    ):
        raise ValueError("num_steps must be a nonnegative integer")
    if (
        isinstance(record_every, bool)
        or not isinstance(record_every, (int, np.integer))
        or record_every < 1
    ):
        raise ValueError("record_every must be a positive integer")
    if not isfinite(step_size) or step_size <= 0:
        raise ValueError("Time step must be finite and positive")

    field = np.array(state, copy=True)
    steps = np.arange(0, int(num_steps) + 1, int(record_every))
    if steps[-1] != num_steps:
        steps = np.append(steps, int(num_steps))
    fields = np.empty((steps.size, grid.num_points), dtype=np.complex128)

    for index, step in enumerate(steps):
        if index:
            field = scheme.advance(field, int(step - steps[index - 1]), step_size)
        if field.shape != grid.theta.shape:
            raise ValueError(f"Field at step {step} must match the spatial grid")
        if not np.isfinite(field).all():
            raise FloatingPointError(f"Non-finite field at step {step}")

        fields[index] = field

    return FieldEvolution(
        theta=grid.theta,
        times=np.asarray(steps * step_size, dtype=np.float64),
        values=fields,
    )
