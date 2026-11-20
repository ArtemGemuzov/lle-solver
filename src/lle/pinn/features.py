"""Представление координат и перенос весов для периодических признаков."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, pi

import torch


@dataclass(frozen=True)
class PeriodicFeatures:
    length: float
    time_scale: float
    harmonics: int

    def __post_init__(self) -> None:
        if (
            not isfinite(self.length)
            or self.length <= 0
            or not isfinite(self.time_scale)
            or self.time_scale <= 0
            or self.harmonics < 1
        ):
            raise ValueError("Feature scales and harmonic count must be positive")

    @property
    def output_size(self) -> int:
        return 1 + 2 * self.harmonics

    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        theta, time = points[:, :1], points[:, 1:2]
        angle = (2 * pi / self.length) * theta
        features = [2 * time / self.time_scale - 1]
        for harmonic in range(1, self.harmonics + 1):
            features.extend((torch.sin(harmonic * angle), torch.cos(harmonic * angle)))
        return torch.cat(features, dim=1)

    def transfer_weights(
        self, weights: Mapping[str, torch.Tensor], target: "PeriodicFeatures"
    ) -> dict[str, torch.Tensor]:
        """Переносит первый слой FNN при смене нормировки временного признака."""
        if self.length != target.length or self.harmonics != target.harmonics:
            raise ValueError("Weight transfer requires matching spatial features")
        if weights["linears.0.weight"].shape[1] != self.output_size:
            raise ValueError("First layer does not match the feature layout")
        return _initial_weights(weights, time_ratio=target.time_scale / self.time_scale)


def _initial_weights(
    weights: Mapping[str, torch.Tensor], *, time_ratio: float
) -> dict[str, torch.Tensor]:
    """Пересчитывает первый слой при замене T на time_ratio * T.

    Исходные тензоры не изменяются; остальные слои сохраняют прежние веса.
    """
    if not isfinite(time_ratio) or time_ratio <= 0:
        raise ValueError("Time ratio must be finite and positive")
    result = dict(weights)
    if time_ratio != 1:
        # tau_old = time_ratio * tau_new + time_ratio - 1.
        weight = weights["linears.0.weight"]
        time_column = weight[:, 0]
        adjusted = weight.clone()
        adjusted[:, 0] = time_ratio * time_column
        result["linears.0.weight"] = adjusted
        result["linears.0.bias"] = (
            weights["linears.0.bias"] + (time_ratio - 1) * time_column
        )
    return result
