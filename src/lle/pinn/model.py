"""Периодическая сеть, начальное условие и невязка LLE."""

from collections.abc import Callable

import deepxde as dde
import numpy as np
import torch
from deepxde.nn.pytorch import FNN
from numpy.typing import NDArray

from lle.physics.equation import LLEParameters
from lle.physics.problem import LLEProblem
from lle.pinn.features import PeriodicFeatures
from lle.pinn.loss.causal import CausalTimePDE
from lle.pinn.loss.relobralo import ReLoBRaLoTimePDE
from lle.pinn.parameters import PinnLossConfig, PinnNetworkConfig


def build_network(
    *, input_size: int, hidden_layers: int, hidden_width: int, output_size: int = 2
) -> FNN:
    """Создаёт FNN; преобразование входа выбирает вызывающий код."""
    return FNN(
        [input_size] + [hidden_width] * hidden_layers + [output_size],
        "tanh",
        "Glorot normal",
    )


def lle_residual(
    equation: LLEParameters,
) -> Callable[[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
    """Возвращает функцию двух компонент невязки LLE."""

    def residual(points, field):
        # Координаты в points — физические theta, t. Компоненты поля — u, v.
        u, v = field[:, :1], field[:, 1:2]
        u_t = dde.grad.jacobian(field, points, i=0, j=1)
        v_t = dde.grad.jacobian(field, points, i=1, j=1)
        u_theta2 = dde.grad.hessian(field, points, component=0, i=0, j=0)
        v_theta2 = dde.grad.hessian(field, points, component=1, i=0, j=0)
        phase = u**2 + v**2 - equation.detuning
        dispersion = equation.beta2 / 2
        return [
            u_t + u + phase * v - dispersion * v_theta2 - equation.pump_amplitude,
            v_t + v - phase * u + dispersion * u_theta2,
        ]

    return residual


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


def build_model(
    problem: LLEProblem,
    *,
    network: PinnNetworkConfig,
    loss: PinnLossConfig,
    interior_points: int,
    initial_points: int,
    seed: int,
) -> dde.Model:
    """Собирает обучающую задачу и выбирает взвешивание потерь."""
    if interior_points < 1 or initial_points < 1:
        raise ValueError("Training point counts must be positive")
    dde.config.set_random_seed(seed)
    half_length = problem.length / 2
    geometry = dde.geometry.GeometryXTime(
        dde.geometry.Interval(-half_length, half_length),
        dde.geometry.TimeDomain(0, problem.time_end),
    )
    residual = lle_residual(problem.equation)
    conditions = [
        dde.icbc.IC(
            geometry,
            lambda points: problem.initial(points[:, 0]).real[:, None],
            lambda _, on: on,
            component=0,
        ),
        dde.icbc.IC(
            geometry,
            lambda points: problem.initial(points[:, 0]).imag[:, None],
            lambda _, on: on,
            component=1,
        ),
    ]
    options = {
        "num_domain": interior_points,
        "num_initial": initial_points,
        "num_test": interior_points,
        "train_distribution": "pseudo",
    }
    balancing = loss.relobralo
    causal = loss.causal
    if balancing is not None:
        data = ReLoBRaLoTimePDE(
            geometry,
            residual,
            conditions,
            **options,
            alpha=balancing.alpha,
            temperature=balancing.temperature,
            lookback_probability=balancing.lookback_probability,
            seed=seed,
        )
    elif causal is not None:
        data = CausalTimePDE(
            geometry,
            residual,
            conditions,
            **options,
            bands=causal.bands,
            epsilon=causal.schedule.epsilons[0] if causal.schedule else causal.epsilon,
            time_end=problem.time_end,
        )
    else:
        data = dde.data.TimePDE(geometry, residual, conditions, **options)
    # Сохраняем порядок генерации точек и инициализации весов для воспроизводимости.
    features = PeriodicFeatures(
        problem.length, problem.time_end, network.spatial_harmonics
    )
    net = build_network(
        input_size=features.output_size,
        hidden_layers=network.hidden_layers,
        hidden_width=network.hidden_width,
    )
    net.apply_feature_transform(features)
    return dde.Model(data, net)
