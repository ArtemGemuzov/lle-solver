"""Эволюция гауссова импульса с помощью PINN."""

from math import pi

import deepxde as dde
import numpy as np
import torch

from lle_runs.config import config_from_cli
from lle_runs.runs import create_run
from lle_solver.grid import PeriodicGrid
from lle_solver.simulations.evolution import FieldEvolution
from lle_solver.simulations.pinn_gaussian import PinnGaussianConfig
from lle_solver.simulations.storage import save_evolution


def build_model(config: PinnGaussianConfig):
    equation = config.equation
    domain = config.domain
    network = config.network
    training = config.training
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
        return config.initial.amplitude * np.exp(-0.5 * (theta / config.initial.width) ** 2)

    def initial_imag(points):
        return np.zeros_like(points[:, :1])

    def on_initial(_, is_initial):
        return is_initial

    data = dde.data.TimePDE(
        geometry, residual,
        [
            dde.icbc.IC(geometry, initial_real, on_initial, component=0),
            dde.icbc.IC(geometry, initial_imag, on_initial, component=1),
        ],
        num_domain=domain.interior_points,
        num_initial=domain.initial_points,
        num_test=domain.interior_points,
        train_distribution="pseudo",
    )

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
    model.compile(
        "adam", lr=training.learning_rate,
        loss_weights=[1.0, 1.0, training.initial_weight, training.initial_weight],
    )
    return model


def predict_evolution(model, config: PinnGaussianConfig) -> FieldEvolution:
    grid = PeriodicGrid(config.recording.num_points, config.domain.length)
    count = round(config.domain.time_end / config.recording.interval)
    times = np.arange(count + 1, dtype=np.float64) * config.recording.interval
    theta, time = np.meshgrid(grid.theta, times)
    points = np.column_stack((theta.ravel(), time.ravel()))
    prediction = np.asarray(model.predict(points), dtype=np.float64)
    values = (prediction[:, 0] + 1j * prediction[:, 1]).reshape(theta.shape)
    if not np.isfinite(values).all():
        raise FloatingPointError("PINN вернула нечисловое поле")
    return FieldEvolution(theta=grid.theta, times=times, values=values)


def main() -> None:
    config = config_from_cli(PinnGaussianConfig)

    training = config.training
    dde.config.set_random_seed(training.seed)
    with create_run(config, script=__file__) as run:
        print(f"Run: {run.id}", flush=True)
        model = build_model(config)
        history, _ = model.train(
            iterations=training.iterations, display_every=training.log_every,
            callbacks=[dde.callbacks.PDEPointResampler(period=training.resample_every)],
        )
        losses = np.column_stack((history.steps, history.loss_train, history.loss_test))
        np.savetxt(
            run.path / "losses.csv", losses, delimiter=",", comments="",
            header=("step,train_pde_u,train_pde_v,train_ic_u,train_ic_v,"
                    "test_pde_u,test_pde_v,test_ic_u,test_ic_v"),
            fmt=["%d"] + ["%.9e"] * 8,
        )
        if not np.isfinite(losses).all():
            raise FloatingPointError("Нечисловое значение функции потерь PINN")
        model.save(str(run.path / "model"), use_iteration_suffix=False)
        save_evolution(run.path, predict_evolution(model, config))


if __name__ == "__main__":
    main()
