"""Параметры построения и обучения PINN."""

from dataclasses import dataclass
from math import isfinite


def _integer(name: str, value: int, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _positive(name: str, value: float) -> None:
    if not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class NetworkParameters:
    hidden_layers: int = 4
    hidden_width: int = 64
    spatial_harmonics: int = 4

    def __post_init__(self) -> None:
        _integer("hidden_layers", self.hidden_layers, 1)
        _integer("hidden_width", self.hidden_width, 1)
        _integer("spatial_harmonics", self.spatial_harmonics, 1)


@dataclass(frozen=True)
class TrainingParameters:
    iterations: int = 20_000
    learning_rate: float = 0.001
    initial_weight: float = 10.0
    resample_every: int = 100
    log_every: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        _integer("iterations", self.iterations, 1)
        _positive("learning_rate", self.learning_rate)
        _positive("initial_weight", self.initial_weight)
        _integer("resample_every", self.resample_every, 1)
        _integer("log_every", self.log_every, 1)
        _integer("seed", self.seed, 0)
        if self.seed >= 2**32:
            raise ValueError("seed must fit in 32 bits")


@dataclass(frozen=True)
class ReLoBRaLoParameters:
    alpha: float = 0.999
    temperature: float = 0.1
    lookback_probability: float = 0.001

    def __post_init__(self) -> None:
        _positive("temperature", self.temperature)
        if not 0 <= self.alpha < 1 or not 0 <= self.lookback_probability <= 1:
            raise ValueError("Invalid ReLoBRaLo weighting parameters")


@dataclass(frozen=True)
class CausalScheduleParameters:
    epsilons: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)
    threshold: float = 0.99

    def __post_init__(self) -> None:
        if len(self.epsilons) < 2 or not 0 < self.threshold < 1:
            raise ValueError("Invalid causal schedule length or threshold")
        for epsilon in self.epsilons:
            _positive("epsilon", epsilon)


@dataclass(frozen=True)
class CausalParameters:
    bands: int = 4
    epsilon: float = 1.0
    schedule: CausalScheduleParameters | None = None

    def __post_init__(self) -> None:
        _integer("bands", self.bands, 2)
        _positive("epsilon", self.epsilon)


@dataclass(frozen=True)
class LossParameters:
    relobralo: ReLoBRaLoParameters | None = None
    causal: CausalParameters | None = None

    def __post_init__(self) -> None:
        if self.relobralo is not None and self.causal is not None:
            raise ValueError("Choose either ReLoBRaLo or causal weighting")


@dataclass(frozen=True)
class RADParameters:
    candidate_points: int = 131072
    replace: bool = True
    batch_size: int = 8192
    power: float = 1.0
    uniform_fraction: float = 0.5
    warmup_steps: int = 5000

    def __post_init__(self) -> None:
        _integer("candidate_points", self.candidate_points, 1)
        _integer("batch_size", self.batch_size, 1)
        _integer("warmup_steps", self.warmup_steps, 0)
        _positive("power", self.power)
        if not isinstance(self.replace, bool):
            raise ValueError("replace must be a boolean")
        if not 0 <= self.uniform_fraction <= 1:
            raise ValueError("uniform_fraction must lie between zero and one")
