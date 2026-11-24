from math import isclose, pi
from typing import Annotated, Self

from pydantic import Field, model_validator

from lle.config import Config
from lle.physics.problem import LLEProblem
from lle.pinn.parameters import (
    CausalParameters,
    CausalScheduleParameters,
    LossParameters,
    ReLoBRaLoParameters,
)
from lle.workflows.parameters import (
    GaussianPulseConfig,
    LLEParametersConfig,
    RecordingConfig,
    gaussian_problem,
)


class PinnNetworkConfig(Config):
    hidden_layers: int = Field(default=4, ge=1)
    hidden_width: int = Field(default=64, ge=1)
    spatial_harmonics: int = Field(default=4, ge=1)


class PinnTrainingConfig(Config):
    iterations: int = Field(default=20_000, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    initial_weight: float = Field(default=10.0, gt=0)
    resample_every: int = Field(default=100, ge=1)
    log_every: int = Field(default=100, ge=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class ReLoBRaLoConfig(Config):
    alpha: float = Field(default=0.999, ge=0, lt=1)
    temperature: float = Field(default=0.1, gt=0)
    lookback_probability: float = Field(default=0.001, ge=0, le=1)


class CausalScheduleConfig(Config):
    epsilons: list[Annotated[float, Field(gt=0)]] = Field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0],
        min_length=2,
    )
    threshold: float = Field(default=0.99, gt=0, lt=1)


class CausalConfig(Config):
    bands: int = Field(default=4, ge=2)
    epsilon: float = Field(default=1.0, gt=0)
    schedule: CausalScheduleConfig | None = None


class PinnLossConfig(Config):
    relobralo: ReLoBRaLoConfig | None = None
    causal: CausalConfig | None = None

    @model_validator(mode="after")
    def validate_weighting(self) -> Self:
        if self.relobralo is not None and self.causal is not None:
            raise ValueError("Choose either ReLoBRaLo or causal weighting")
        return self


class RADConfig(Config):
    candidate_points: int = Field(default=131072, ge=1)
    replace: bool = True
    batch_size: int = Field(default=8192, ge=1)
    power: float = Field(default=1.0, gt=0)
    uniform_fraction: float = Field(default=0.5, ge=0, le=1)
    warmup_steps: int = Field(default=5000, ge=0)


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

    def loss_parameters(self) -> LossParameters:
        causal = self.loss.causal
        balancing = self.loss.relobralo
        schedule = causal.schedule if causal else None
        weighting = (
            CausalParameters(
                bands=causal.bands,
                epsilon=causal.epsilon,
                schedule=CausalScheduleParameters(
                    tuple(schedule.epsilons), schedule.threshold
                )
                if schedule
                else None,
            )
            if causal
            else None
        )
        return LossParameters(
            causal=weighting,
            relobralo=ReLoBRaLoParameters(**balancing.model_dump())
            if balancing
            else None,
        )

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
    def validate_method(self) -> Self:
        rad = self.sampling.rad
        if rad and rad.candidate_points < self.domain.interior_points:
            raise ValueError("RAD candidate_points must be at least interior_points")
        if rad and (self.loss.causal or self.loss.relobralo):
            raise ValueError("RAD currently requires fixed loss weights")
        causal = self.loss.causal
        if (
            causal
            and causal.schedule
            and self.training.iterations < len(causal.schedule.epsilons)
        ):
            raise ValueError(
                "Each causal weighting stage requires at least one Adam step"
            )
        return self
