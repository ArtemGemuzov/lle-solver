from typing import Self

import numpy as np
from pydantic import Field, model_validator

from lle_runs.config import Config
from lle_solver.simulations.configs import EquationConfig, GridConfig, GaussianPulseConfig, SolverConfig, RecordingConfig


class GaussianConfig(Config):
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
                raise ValueError("Времена должны быть положительными кратными dt")
            counts.append(count)

        total, interval = counts
        if total % interval:
            raise ValueError("Время моделирования должно быть кратно интервалу записи")

        return self
