"""Запись результатов обучения в каталог запуска."""

import csv
import platform

import deepxde as dde
import numpy as np
import torch
from deepxde.model import LossHistory

from lle.fields import FieldEvolution
from lle.pinn.loss import CausalStage
from lle.pinn.training import PinnTraining, predict_evolution
from lle.runs.runs import Run
from lle.storage import save_evolution

from .config import PinnGaussianConfig


def predict(model: dde.Model, config: PinnGaussianConfig) -> FieldEvolution:
    return predict_evolution(
        model.net,
        length=config.domain.length,
        time_end=config.domain.time_end,
        interval=config.recording.interval,
        num_points=config.recording.num_points,
    )


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
    run: Run, config: PinnGaussianConfig, training: PinnTraining, history: LossHistory
) -> None:
    model = training.model
    schedule = training.checkpoint.schedule
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
    if schedule is not None:
        with (run.path / "causal_stages.csv").open(
            "x", encoding="utf-8", newline=""
        ) as target:
            writer = csv.writer(target)
            writer.writerow(CausalStage._fields)
            writer.writerows(schedule.history)
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
