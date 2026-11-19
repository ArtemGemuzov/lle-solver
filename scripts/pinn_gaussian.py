"""Эволюция гауссова импульса с помощью PINN."""

import argparse
import csv
from math import pi
from pathlib import Path
import platform
import shutil

import deepxde as dde
import numpy as np
import torch
import yaml

from lle_runs.config import load_config, parse_config_args
from lle_runs.runs import create_run, resume_run
from lle_solver.grid import PeriodicGrid
from lle_solver.simulations.evolution import FieldEvolution
from lle_solver.simulations.pinn_gaussian import PinnGaussianConfig
from lle_solver.simulations.pinn_checkpoint import PinnCheckpoint, extend_checkpoint, load_checkpoint
from lle_solver.simulations.pinn_loss import CausalSchedule, CausalStage, CausalTimePDE, ReLoBRaLoTimePDE
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

    initial_conditions = [
        dde.icbc.IC(geometry, initial_real, on_initial, component=0),
        dde.icbc.IC(geometry, initial_imag, on_initial, component=1),
    ]
    data_options = dict(
        num_domain=domain.interior_points,
        num_initial=domain.initial_points,
        num_test=domain.interior_points,
        train_distribution="pseudo",
    )
    balancing = config.loss.relobralo
    causal = config.loss.causal
    if balancing is not None:
        data = ReLoBRaLoTimePDE(
            geometry, residual, initial_conditions, **data_options,
            balancing=balancing, seed=training.seed,
        )
    elif causal is not None:
        data = CausalTimePDE(
            geometry, residual, initial_conditions, **data_options,
            causal=causal, time_end=domain.time_end,
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
        "adam", lr=training.learning_rate,
        loss_weights=[1.0, 1.0, initial_weight, initial_weight],
    )
    if balancing is not None or causal is not None:
        model.opt.register_step_pre_hook(data.before_optimizer_step)
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


def training_runtime(device: torch.device) -> dict[str, str]:
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    elif device.type == "cpu":
        device_name = platform.processor() or platform.machine()
    else:
        device_name = device.type
    return {
        "device": str(device),
        "device_name": device_name,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", type=Path, help="Продолжить обучение из checkpoint.pt")
    modes.add_argument("--extend-from", type=Path, help="Начать дочерний Run из checkpoint.pt")
    parser.add_argument("--extra-steps", type=int, help="Число дополнительных шагов Adam")
    args = parse_config_args(parser)
    if args.extend_from is not None:
        if args.extra_steps is None or args.extra_steps < 1:
            parser.error("--extend-from требует положительного --extra-steps")
    elif args.extra_steps is not None:
        parser.error("--extra-steps используется только с --extend-from")
    source = args.resume or args.extend_from
    if source is not None:
        if source.name != "checkpoint.pt":
            parser.error("Укажите файл checkpoint.pt исходного запуска")
        if args.config is not None or args.overrides:
            parser.error("Конфигурация берётся из чекпоинта целиком")
    try:
        config = (None if source is not None
                  else load_config(PinnGaussianConfig, args.config, args.overrides))
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    return config, args


def clear_resumed_outputs(path: Path, completed_stages: int) -> None:
    """Discard exports beyond the checkpoint; retain completed stage exports."""
    for name in ("losses.csv", "loss_weights.csv", "causal_stages.csv",
                 "model.pt", "field_evolution.npz"):
        (path / name).unlink(missing_ok=True)
    for stage in path.glob("stage_*"):
        suffix = stage.name.removeprefix("stage_")
        if stage.is_dir() and suffix.isdigit() and int(suffix) > completed_stages:
            shutil.rmtree(stage)


def main() -> None:
    config, args = parse_args()
    saved_checkpoint = None
    if args.extend_from is not None:
        source = args.extend_from.resolve()
        saved_checkpoint = extend_checkpoint(load_checkpoint(source), args.extra_steps)
        config = PinnGaussianConfig.model_validate(saved_checkpoint["config"])
        context = create_run(config, script=__file__, extra_metadata={
            "parent_run_id": source.parent.name,
            "parent_step": saved_checkpoint["train_state"]["step"],
        })
    elif args.resume is not None:
        context = resume_run(args.resume.parent, script=__file__)
    else:
        context = create_run(config, script=__file__)
    with context as run:
        if args.resume is not None:
            # Read only after acquiring the lock: another process may replace it.
            saved_checkpoint = load_checkpoint(run.path / "checkpoint.pt")
            config = PinnGaussianConfig.model_validate(saved_checkpoint["config"])
            if config != run.load_config(PinnGaussianConfig):
                raise ValueError("Checkpoint configuration differs from the run")
        training = config.training
        dde.config.set_random_seed(training.seed)
        runtime = training_runtime(torch.get_default_device())
        metadata = run.load_metadata()
        metadata["runtime"] = runtime
        run.save_metadata(metadata)
        print(f"Run: {run.id}", flush=True)
        print(f"Device: {runtime['device']} ({runtime['device_name']})", flush=True)
        model = build_model(config)
        resampler = dde.callbacks.PDEPointResampler(period=training.resample_every)
        callbacks = [resampler]
        causal = config.loss.causal
        schedule = None
        if causal is not None and causal.schedule is not None:
            def save_stage(stage: CausalStage) -> None:
                stage_dir = run.path / f"stage_{stage.number:02d}"
                stage_dir.mkdir()
                save_evolution(stage_dir, predict_evolution(model, config))
                print(f"Stage {stage.number}: epsilon={stage.epsilon:g}, step={stage.last_step}, "
                      f"min_weight={stage.min_weight:.6f}, {stage.reason}", flush=True)

            stage_limit = training.iterations // len(causal.schedule.epsilons)
            schedule = CausalSchedule(causal.schedule, stage_limit, save_stage)
            callbacks.append(schedule)

        checkpoint = PinnCheckpoint(run.path / "checkpoint.pt", config, resampler, schedule)
        checkpoint.set_model(model)
        if saved_checkpoint is not None:
            checkpoint.restore(saved_checkpoint)
            if args.extend_from is not None:
                # Allow resuming the child even before its first periodic save.
                checkpoint.save()
            else:
                clear_resumed_outputs(run.path, len(schedule.history) if schedule else 0)
            print(f"Resumed at step {model.train_state.step}", flush=True)
        callbacks.append(checkpoint)
        remaining = 0 if checkpoint.complete else training.iterations - model.train_state.step
        history, state = model.train(
            iterations=remaining, display_every=training.log_every,
            callbacks=callbacks,
        )
        if schedule is not None and history.steps[-1] != state.step:
            history, _ = model.train(iterations=0, verbose=0)
            checkpoint.save()
        losses = np.column_stack((history.steps, history.loss_train, history.loss_test))
        np.savetxt(
            run.path / "losses.csv", losses, delimiter=",", comments="",
            header=("step,train_pde_u,train_pde_v,train_ic_u,train_ic_v,"
                    "test_pde_u,test_pde_v,test_ic_u,test_ic_v"),
            fmt=["%d"] + ["%.9e"] * 8,
        )
        if not np.isfinite(losses).all():
            raise FloatingPointError("Нечисловое значение функции потерь PINN")
        if schedule is not None:
            with (run.path / "causal_stages.csv").open("x", encoding="utf-8", newline="") as target:
                writer = csv.writer(target)
                writer.writerow(CausalStage._fields)
                writer.writerows(schedule.history)
        if config.loss.relobralo is not None or causal is not None:
            if causal is None:
                columns = ["raw_pde", "raw_ic", "weight_pde", "weight_ic"]
            else:
                bands = range(1, causal.bands + 1)
                columns = [f"raw_band_{i}" for i in bands] + [f"weight_band_{i}" for i in bands]
            weight_history = torch.stack(model.data.weight_history).cpu().numpy()
            steps = np.arange(1, len(weight_history) + 1)
            np.savetxt(
                run.path / "loss_weights.csv",
                np.column_stack((steps, weight_history)),
                delimiter=",", comments="",
                header=",".join(["step", *columns]),
                fmt=["%d"] + ["%.9e"] * len(columns),
            )
        model.save(str(run.path / "model"), use_iteration_suffix=False)
        save_evolution(run.path, predict_evolution(model, config))


if __name__ == "__main__":
    main()
