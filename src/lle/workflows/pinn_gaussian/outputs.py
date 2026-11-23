"""Запись результатов обучения в каталог запуска."""

import csv
import platform

import deepxde as dde
import numpy as np
import torch
from deepxde.model import LossHistory

from lle.physics.fields import FieldEvolution
from lle.physics.geometry import periodic_coordinates
from lle.pinn.loss.schedule import CausalStage
from lle.pinn.training import predict_evolution
from lle.runs.checkpoint import save_checkpoint
from lle.runs.runs import Run
from lle.storage import save_evolution

from .config import PinnGaussianConfig


def predict(model: dde.Model, config: PinnGaussianConfig) -> FieldEvolution:
    count = round(config.domain.time_end / config.recording.interval)
    times = np.arange(count + 1, dtype=np.float64) * config.recording.interval
    theta = periodic_coordinates(config.recording.num_points, config.domain.length)
    return predict_evolution(model.net, theta=theta, times=times)


def save_stage(
    run: Run, config: PinnGaussianConfig, model: dde.Model, stage: CausalStage
) -> None:
    stage_dir = run.path / f"stage_{stage.number:02d}"
    stage_dir.mkdir()
    save_evolution(stage_dir, predict(model, config))
    print(
        f"Стадия {stage.number}: ε={stage.epsilon:g}, шаг={stage.last_step}, "
        f"минимальный вес={stage.min_weight:.6f}, {stage.reason}",
        flush=True,
    )


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


def save_results(
    run: Run, config: PinnGaussianConfig, model: dde.Model, history: LossHistory
) -> None:
    causal = config.loss.causal
    losses = np.column_stack((history.steps, history.loss_train, history.loss_test))
    np.savetxt(
        run.path / "losses.csv",
        losses,
        delimiter=",",
        comments="",
        header=(
            "step,train_pde_u,train_pde_v,train_ic_u,train_ic_v,"
            "test_pde_u,test_pde_v,test_ic_u,test_ic_v"
        ),
        fmt=["%d"] + ["%.9e"] * 8,
    )
    if not np.isfinite(losses).all():
        raise FloatingPointError("Non-finite PINN loss detected")
    if config.loss.relobralo is not None or causal is not None:
        if causal is None:
            columns = ["raw_pde", "raw_ic", "weight_pde", "weight_ic"]
        else:
            bands = range(1, causal.bands + 1)
            columns = [f"raw_band_{i}" for i in bands] + [
                f"weight_band_{i}" for i in bands
            ]
        weight_history = torch.stack(model.data.weight_history).cpu().numpy()
        steps = np.arange(1, len(weight_history) + 1)
        np.savetxt(
            run.path / "loss_weights.csv",
            np.column_stack((steps, weight_history)),
            delimiter=",",
            comments="",
            header=",".join(["step", *columns]),
            fmt=["%d"] + ["%.9e"] * len(columns),
        )
    model.save(str(run.path / "model"), use_iteration_suffix=False)
    save_evolution(run.path, predict(model, config))


def save_training_checkpoint(run: Run, config: PinnGaussianConfig, state: dict) -> None:
    """Сохраняет снимок обучения и диагностику адаптивных точек."""
    save_checkpoint(run.path / "checkpoint.pt", state, config.model_dump(mode="json"))
    schedule = state["schedule"]
    if schedule is not None:
        with (run.path / "causal_stages.csv").open(
            "w", encoding="utf-8", newline=""
        ) as target:
            writer = csv.writer(target)
            writer.writerow(CausalStage._fields)
            writer.writerows(schedule["history"])
    sampling = state.get("sampling")
    if sampling is None:
        return
    columns = (
        "step",
        "residual_mean",
        "residual_max",
        "residual_p90",
        "unique_points",
        "seconds",
    )
    with (run.path / "sampling.csv").open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sampling["history"])
    np.savez_compressed(
        run.path / "sampling_histograms.npz",
        steps=np.array([row["step"] for row in sampling["history"]], dtype=np.int64),
        counts=np.array(
            [row["histogram"] for row in sampling["history"]], dtype=np.int64
        ).reshape(-1, 16, 16),
        theta_edges=np.linspace(
            -config.domain.length / 2, config.domain.length / 2, 17
        ),
        time_edges=np.linspace(0, config.domain.time_end, 17),
    )
