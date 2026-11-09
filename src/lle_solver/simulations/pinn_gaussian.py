from math import isclose, pi
from typing import Annotated, Self

from pydantic import Field, model_validator

from lle_runs.config import Config
from lle_solver.simulations.configs import EquationConfig, GaussianPulseConfig, RecordingConfig


class PinnDomainConfig(Config):
    length: float = Field(default=2.0 * pi, gt=0)
    time_end: float = Field(default=30.0, gt=0)
    interior_points: int = Field(default=1024, ge=1)
    initial_points: int = Field(default=256, ge=1)


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
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0], min_length=2,
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
            raise ValueError("Выберите один способ взвешивания: ReLoBRaLo или причинные веса")
        return self


class PinnRecordingConfig(RecordingConfig):
    num_points: int = Field(default=512, ge=2)


class PinnGaussianConfig(Config):
    equation: EquationConfig = Field(default_factory=EquationConfig)
    initial: GaussianPulseConfig = Field(default_factory=GaussianPulseConfig)
    domain: PinnDomainConfig = Field(default_factory=PinnDomainConfig)
    network: PinnNetworkConfig = Field(default_factory=PinnNetworkConfig)
    training: PinnTrainingConfig = Field(default_factory=PinnTrainingConfig)
    recording: PinnRecordingConfig = Field(default_factory=PinnRecordingConfig)
    loss: PinnLossConfig = Field(default_factory=PinnLossConfig)

    @model_validator(mode="after")
    def validate_recording(self) -> Self:
        time_end = self.domain.time_end
        interval = self.recording.interval
        count = round(time_end / interval)
        if count < 1 or not isclose(time_end, count * interval, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Конечное время должно быть кратно интервалу записи")
        return self

    @model_validator(mode="after")
    def validate_schedule(self) -> Self:
        causal = self.loss.causal
        if causal is not None and causal.schedule is not None:
            if self.training.iterations < len(causal.schedule.epsilons):
                raise ValueError("Для каждой стадии причинного взвешивания нужен хотя бы один шаг Adam")
        return self
