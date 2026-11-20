"""Сеть, периодические признаки и невязка LLE для гауссова импульса."""

from math import pi

import deepxde as dde
import numpy as np
import torch

from lle.pinn.loss import CausalTimePDE, ReLoBRaLoTimePDE


def build_model(*, equation, initial, domain, network, training, loss):
    half_length = domain.length / 2
    geometry = dde.geometry.GeometryXTime(
        dde.geometry.Interval(-half_length, half_length),
        dde.geometry.TimeDomain(0, domain.time_end),
    )

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

    def initial_real(points):
        theta = points[:, :1]
        return initial.amplitude * np.exp(-0.5 * (theta / initial.width) ** 2)

    def initial_imag(points):
        return np.zeros_like(points[:, :1])

    def on_initial(_, is_initial):
        return is_initial

    initial_conditions = [
        dde.icbc.IC(geometry, initial_real, on_initial, component=0),
        dde.icbc.IC(geometry, initial_imag, on_initial, component=1),
    ]
    data_options = {
        "num_domain": domain.interior_points,
        "num_initial": domain.initial_points,
        "num_test": domain.interior_points,
        "train_distribution": "pseudo",
    }
    balancing = loss.relobralo
    causal = loss.causal
    if balancing is not None:
        data = ReLoBRaLoTimePDE(
            geometry,
            residual,
            initial_conditions,
            **data_options,
            balancing=balancing,
            seed=training.seed,
        )
    elif causal is not None:
        data = CausalTimePDE(
            geometry,
            residual,
            initial_conditions,
            **data_options,
            causal=causal,
            time_end=domain.time_end,
        )
    else:
        data = dde.data.TimePDE(geometry, residual, initial_conditions, **data_options)

    def periodic_features(points):
        theta, time = points[:, :1], points[:, 1:2]
        angle = (pi / half_length) * theta
        features = [2 * time / domain.time_end - 1]
        for harmonic in range(1, network.spatial_harmonics + 1):
            features.extend((torch.sin(harmonic * angle), torch.cos(harmonic * angle)))
        return torch.cat(features, dim=1)

    layer_sizes = [1 + 2 * network.spatial_harmonics]
    layer_sizes += [network.hidden_width] * network.hidden_layers + [2]
    net = dde.nn.FNN(layer_sizes, "tanh", "Glorot normal")
    net.apply_feature_transform(periodic_features)
    model = dde.Model(data, net)
    initial_weight = training.initial_weight if balancing is None else 1.0
    model.compile(
        "adam",
        lr=training.learning_rate,
        loss_weights=[1.0, 1.0, initial_weight, initial_weight],
    )
    if balancing is not None or causal is not None:
        model.opt.register_step_pre_hook(data.before_optimizer_step)
    return model
