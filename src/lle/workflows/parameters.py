from math import pi

from pydantic import Field

from lle.config import Config
from lle.physics.equation import LLEParameters
from lle.physics.initial import GaussianPulse
from lle.physics.problem import LLEProblem


class LLEParametersConfig(Config):
    pump_amplitude: float = 2.0
    detuning: float = 3.0
    beta2: float = -0.1


class GridConfig(Config):
    length: float = Field(default=2.0 * pi, gt=0)
    num_points: int = Field(default=512, ge=2)


class GaussianPulseConfig(Config):
    amplitude: float = 1.8
    width: float = Field(default=0.3, gt=0)


class SolverConfig(Config):
    time_step: float = Field(default=5e-4, gt=0)
    time_end: float = Field(default=100.0, gt=0)


class RecordingConfig(Config):
    interval: float = Field(default=0.5, gt=0)


def gaussian_problem(
    equation: LLEParametersConfig,
    initial: GaussianPulseConfig,
    *,
    length: float,
    time_end: float,
) -> LLEProblem:
    return LLEProblem(
        equation=LLEParameters(**equation.model_dump()),
        initial=GaussianPulse(**initial.model_dump()),
        length=length,
        time_end=time_end,
    )
