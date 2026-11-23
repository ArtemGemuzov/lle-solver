from typing import Self

from pydantic import Field, model_validator

from lle.config import Config
from lle.numerics.ssfm import recording_steps
from lle.physics.problem import LLEProblem
from lle.workflows.parameters import (
    GaussianPulseConfig,
    GridConfig,
    LLEParametersConfig,
    RecordingConfig,
    SolverConfig,
    gaussian_problem,
)


class SsfmGaussianConfig(Config):
    equation: LLEParametersConfig = Field(default_factory=LLEParametersConfig)
    grid: GridConfig = Field(default_factory=GridConfig)
    initial: GaussianPulseConfig = Field(default_factory=GaussianPulseConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        recording_steps(
            self.solver.time_end, self.recording.interval, self.solver.time_step
        )
        return self

    def to_problem(self) -> LLEProblem:
        return gaussian_problem(
            self.equation,
            self.initial,
            length=self.grid.length,
            time_end=self.solver.time_end,
        )
