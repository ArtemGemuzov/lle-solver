from math import isclose, pi
from typing import Self

from pydantic import Field, model_validator

from lle.config import Config
from lle.physics.problem import LLEProblem
from lle.pinn.parameters import (
    PinnLossConfig,
    PinnNetworkConfig,
    PinnTrainingConfig,
    RADConfig,
)
from lle.workflows.parameters import (
    GaussianPulseConfig,
    LLEParametersConfig,
    RecordingConfig,
    gaussian_problem,
)


class PinnDomainConfig(Config):
    length: float = Field(default=2.0 * pi, gt=0)
    time_end: float = Field(default=30.0, gt=0)
    interior_points: int = Field(default=1024, ge=1)
    initial_points: int = Field(default=256, ge=1)


class PinnRecordingConfig(RecordingConfig):
    num_points: int = Field(default=512, ge=2)


class PinnCheckpointConfig(Config):
    every_steps: int = Field(default=1000, ge=1)


class PinnSamplingConfig(Config):
    rad: RADConfig | None = None


class PinnGaussianConfig(Config):
    equation: LLEParametersConfig = Field(default_factory=LLEParametersConfig)
    initial: GaussianPulseConfig = Field(default_factory=GaussianPulseConfig)
    domain: PinnDomainConfig = Field(default_factory=PinnDomainConfig)
    network: PinnNetworkConfig = Field(default_factory=PinnNetworkConfig)
    training: PinnTrainingConfig = Field(default_factory=PinnTrainingConfig)
    recording: PinnRecordingConfig = Field(default_factory=PinnRecordingConfig)
    loss: PinnLossConfig = Field(default_factory=PinnLossConfig)
    checkpoint: PinnCheckpointConfig = Field(default_factory=PinnCheckpointConfig)
    sampling: PinnSamplingConfig = Field(default_factory=PinnSamplingConfig)

    def to_problem(self) -> LLEProblem:
        return gaussian_problem(
            self.equation,
            self.initial,
            length=self.domain.length,
            time_end=self.domain.time_end,
        )

    @model_validator(mode="after")
    def validate_sampling(self) -> Self:
        rad = self.sampling.rad
        if rad is not None:
            if rad.candidate_points < self.domain.interior_points:
                raise ValueError(
                    "RAD candidate_points must be at least interior_points"
                )
            if self.loss.causal is not None or self.loss.relobralo is not None:
                raise ValueError("RAD currently requires fixed loss weights")
        return self

    @model_validator(mode="after")
    def validate_recording(self) -> Self:
        time_end = self.domain.time_end
        interval = self.recording.interval
        count = round(time_end / interval)
        if count < 1 or not isclose(
            time_end, count * interval, rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError("End time must be a multiple of the recording interval")
        return self

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        causal = self.loss.causal
        if (
            causal is not None
            and causal.schedule is not None
            and self.training.iterations < len(causal.schedule.epsilons)
        ):
            raise ValueError(
                "Each causal weighting stage requires at least one Adam step"
            )
        return self
