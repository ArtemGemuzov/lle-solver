"""Параметры построения и обучения PINN."""

from typing import Annotated, Self

from pydantic import BaseModel, Field, model_validator


class _PinnSettings(
    BaseModel,
    extra="forbid",
    strict=True,
    frozen=True,
    allow_inf_nan=False,
    validate_default=True,
):
    pass


class PinnNetworkConfig(_PinnSettings):
    hidden_layers: int = Field(default=4, ge=1)
    hidden_width: int = Field(default=64, ge=1)
    spatial_harmonics: int = Field(default=4, ge=1)


class PinnTrainingConfig(_PinnSettings):
    iterations: int = Field(default=20_000, ge=1)
    learning_rate: float = Field(default=0.001, gt=0)
    initial_weight: float = Field(default=10.0, gt=0)
    resample_every: int = Field(default=100, ge=1)
    log_every: int = Field(default=100, ge=1)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class ReLoBRaLoConfig(_PinnSettings):
    alpha: float = Field(default=0.999, ge=0, lt=1)
    temperature: float = Field(default=0.1, gt=0)
    lookback_probability: float = Field(default=0.001, ge=0, le=1)


class CausalScheduleConfig(_PinnSettings):
    epsilons: list[Annotated[float, Field(gt=0)]] = Field(
        default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0],
        min_length=2,
    )
    threshold: float = Field(default=0.99, gt=0, lt=1)


class CausalConfig(_PinnSettings):
    bands: int = Field(default=4, ge=2)
    epsilon: float = Field(default=1.0, gt=0)
    schedule: CausalScheduleConfig | None = None


class PinnLossConfig(_PinnSettings):
    relobralo: ReLoBRaLoConfig | None = None
    causal: CausalConfig | None = None

    @model_validator(mode="after")
    def validate_weighting(self) -> Self:
        if self.relobralo is not None and self.causal is not None:
            raise ValueError("Choose either ReLoBRaLo or causal weighting")
        return self


class RADConfig(_PinnSettings):
    candidate_points: int = Field(default=131072, ge=1)
    replace: bool = True
    batch_size: int = Field(default=8192, ge=1)
    power: float = Field(default=1.0, gt=0)
    uniform_fraction: float = Field(default=0.5, ge=0, le=1)
    warmup_steps: int = Field(default=5000, ge=0)
