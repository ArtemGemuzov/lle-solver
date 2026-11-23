from functools import partial
from math import isfinite

import numpy as np

from lle.numerics.grid import PeriodicGrid
from lle.numerics.integrators import rk4_step
from lle.numerics.lle import linear_operator, nonlinear_term
from lle.numerics.split_schemes import StrangSplitStep
from lle.numerics.steps import FourierStep, IntegratorStep
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


def solve(
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

    field = problem.initial(grid.theta)
    if field.shape != grid.theta.shape:
        raise ValueError("Initial field must match the spatial grid")

    steps = np.arange(0, total + 1, record_step)
    fields = np.empty((steps.size, grid.num_points), dtype=np.complex128)

    for index, step in enumerate(steps):
        if index:
            field = scheme.advance(field, record_step, dt)

        if not np.isfinite(field).all():
            raise FloatingPointError(f"Non-finite field at step {step}")

        fields[index] = field

    return FieldEvolution(
        theta=grid.theta,
        times=np.asarray(steps * dt, dtype=np.float64),
        values=fields,
    )
