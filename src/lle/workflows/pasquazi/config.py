from typing import Self

from pydantic import Field, model_validator

from lle.config import Config
from lle.numerics.ssfm import recording_steps
from lle.physics.pasquazi import PasquaziParameters
from lle.workflows.parameters import GaussianPulseConfig, RecordingConfig, SolverConfig


class EquationConfig(Config):
    n: int = Field(default=PasquaziParameters.n, ge=0)
    zeta_a: float = PasquaziParameters.zeta_a
    zeta_b: float = PasquaziParameters.zeta_b
    sigma: float = Field(default=PasquaziParameters.sigma, ge=0)
    kappa: float = Field(default=PasquaziParameters.kappa, ge=0)
    gain: float = PasquaziParameters.gain
    detuning: float = PasquaziParameters.detuning
    fsr_mismatch: float = PasquaziParameters.fsr_mismatch


class PasquaziConfig(Config):
    equation: EquationConfig = Field(default_factory=EquationConfig)
    num_points: int = Field(default=256, ge=2)
    initial: GaussianPulseConfig = Field(
        default_factory=lambda: GaussianPulseConfig(width=0.03)
    )
    solver: SolverConfig = Field(
        default_factory=lambda: SolverConfig(time_step=0.001, time_end=1.0)
    )
    recording: RecordingConfig = Field(
        default_factory=lambda: RecordingConfig(interval=0.1)
    )

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        recording_steps(
            self.solver.time_end, self.recording.interval, self.solver.time_step
        )
        return self
