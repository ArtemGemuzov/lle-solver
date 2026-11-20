"""Периодическая сеть, начальное условие и невязка LLE."""

from collections.abc import Callable, Mapping
from math import isfinite, pi

import deepxde as dde
import numpy as np
import torch
from deepxde.nn.pytorch import FNN
from numpy.typing import NDArray


def build_network(
    *,
    hidden_layers: int,
    hidden_width: int,
    spatial_harmonics: int,
    length: float,
    time_end: float,
) -> FNN:
    """Создаёт сеть без обучающих точек, начальных условий и оптимизатора."""
    half_length = length / 2

    def periodic_features(points):
        theta, time = points[:, :1], points[:, 1:2]
        angle = (pi / half_length) * theta
        features = [2 * time / time_end - 1]
        for harmonic in range(1, spatial_harmonics + 1):
            features.extend((torch.sin(harmonic * angle), torch.cos(harmonic * angle)))
        return torch.cat(features, dim=1)

    layer_sizes = [1 + 2 * spatial_harmonics]
    layer_sizes += [hidden_width] * hidden_layers + [2]
    net = FNN(layer_sizes, "tanh", "Glorot normal")
    net.apply_feature_transform(periodic_features)
    return net


def lle_residual(
    *, pump_amplitude: float, detuning: float, beta2: float
) -> Callable[[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
    """Возвращает функцию двух компонент невязки LLE."""

    def residual(points, field):
        # Координаты в points — физические theta, t. Компоненты поля — u, v.
        u, v = field[:, :1], field[:, 1:2]
        u_t = dde.grad.jacobian(field, points, i=0, j=1)
        v_t = dde.grad.jacobian(field, points, i=1, j=1)
        u_theta2 = dde.grad.hessian(field, points, component=0, i=0, j=0)
        v_theta2 = dde.grad.hessian(field, points, component=1, i=0, j=0)
        phase = u**2 + v**2 - detuning
        dispersion = beta2 / 2
        return [
            u_t + u + phase * v - dispersion * v_theta2 - pump_amplitude,
            v_t + v - phase * u + dispersion * u_theta2,
        ]

    return residual


def gaussian_initial(
    *, amplitude: float, width: float
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Возвращает действительную и мнимую части гауссова поля в двух столбцах."""

    def initial(points: NDArray[np.floating]) -> NDArray[np.floating]:
        real = amplitude * np.exp(-0.5 * (points[:, :1] / width) ** 2)
        return np.column_stack((real, np.zeros_like(real)))

    return initial


def evaluate_residual(
    net: torch.nn.Module,
    points: NDArray[np.floating],
    residual: Callable[[torch.Tensor, torch.Tensor], list[torch.Tensor]],
) -> list[NDArray[np.floating]]:
    """Вычисляет невязку в заданных точках без обучающей задачи и Adam."""
    net.eval()
    parameter = next(net.parameters())
    try:
        with torch.enable_grad():
            inputs = torch.as_tensor(
                points, dtype=parameter.dtype, device=parameter.device
            ).requires_grad_()
            return [
                value.detach().cpu().numpy() for value in residual(inputs, net(inputs))
            ]
    finally:
        dde.grad.clear()


def predict_field(
    net: torch.nn.Module, points: NDArray[np.floating]
) -> NDArray[np.complex128]:
    """Предсказывает комплексное поле в точках (theta, t) без оптимизатора."""
    net.eval()
    parameter = next(net.parameters())
    with torch.no_grad():
        inputs = torch.as_tensor(points, dtype=parameter.dtype, device=parameter.device)
        values = net(inputs).detach().cpu().numpy().astype(np.float64)
    return values[:, 0] + 1j * values[:, 1]


def initial_weights(
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
