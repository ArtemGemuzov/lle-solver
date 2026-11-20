from functools import partial
from typing import Self

import numpy as np
from pydantic import Field, model_validator

from lle.config import Config
from lle.numerics.grid import PeriodicGrid
from lle.numerics.integrators import rk4_step
from lle.numerics.lle import linear_operator, nonlinear_term
from lle.numerics.split_schemes import StrangSplitStep
from lle.numerics.steps import FourierStep, IntegratorStep
from lle.physics.equation import LLEParameters
from lle.physics.fields import FieldEvolution
from lle.physics.initial import GaussianPulse
from lle.workflows.parameters import (
    EquationConfig,
    GaussianPulseConfig,
    GridConfig,
    RecordingConfig,
    SolverConfig,
)


class SsfmGaussianConfig(Config):
    equation: EquationConfig = Field(default_factory=EquationConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    initial: GaussianPulseConfig = Field(default_factory=GaussianPulseConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        dt = self.solver.time_step
        counts = []

        for duration in (self.solver.time_end, self.recording.interval):
            count = round(duration / dt)
            if count < 1 or not np.isclose(count * dt, duration, rtol=0, atol=1e-12):
                raise ValueError("Times must be positive multiples of dt")
            counts.append(count)

        total, interval = counts
        if total % interval:
            raise ValueError("End time must be a multiple of the recording interval")

        return self


def simulate(p: SsfmGaussianConfig) -> FieldEvolution:
    """Вычисляет эволюцию гауссова импульса без создания Run и записи на диск."""
    dt = p.solver.time_step

    total = round(p.solver.time_end / dt)
    record_step = round(p.recording.interval / dt)

    grid = PeriodicGrid(p.grid.num_points, p.grid.length)
    equation = LLEParameters(**p.equation.model_dump())
    scheme = StrangSplitStep(
        FourierStep(partial(linear_operator, equation), grid.wavenumbers),
        IntegratorStep(partial(nonlinear_term, equation), rk4_step),
    )

    field = GaussianPulse(**p.initial.model_dump())(grid.theta)

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
