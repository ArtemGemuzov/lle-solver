"""Запись результатов обучения в каталог запуска."""

import csv
import platform

import deepxde as dde
import numpy as np
import torch
from deepxde.model import LossHistory

from lle.physics.fields import FieldEvolution
from lle.pinn.loss.schedule import CausalStage
from lle.pinn.sampling import RADSampler
from lle.pinn.training import PinnTraining, predict_evolution
from lle.runs.checkpoint import save_checkpoint
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
    save_sampling(run, training)
    schedule = training.schedule
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


def save_sampling(run: Run, training: PinnTraining) -> None:
    """Сохраняет статистики RAD и распределение точек по одинаковым ячейкам."""
    sampler = training.state.resampler
    if not isinstance(sampler, RADSampler):
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
        writer.writerows(sampler.history)
    np.savez_compressed(
        run.path / "sampling_histograms.npz",
        steps=np.array([row["step"] for row in sampler.history], dtype=np.int64),
        counts=np.array(
            [row["histogram"] for row in sampler.history], dtype=np.int64
        ).reshape(-1, 16, 16),
        theta_edges=np.linspace(-sampler.length / 2, sampler.length / 2, 17),
        time_edges=np.linspace(0, sampler.time_end, 17),
    )


class TrainingCheckpoint(dde.callbacks.Callback):
    """Организует периодическое сохранение состояния выбранного Run."""

    def __init__(self, run: Run, config: PinnGaussianConfig, training: PinnTraining):
        super().__init__()
        self.run = run
        self.config = config
        self.training = training
        self._stage_count = 0
        self._last_saved_step = None

    def save(self) -> None:
        save_checkpoint(
            self.run.path / "checkpoint.pt",
            self.training.state_dict(),
            self.config.model_dump(mode="json"),
        )
        self._last_saved_step = self.model.train_state.step
        save_sampling(self.run, self.training)

    def on_train_begin(self) -> None:
        schedule = self.training.schedule
        self._stage_count = len(schedule.history) if schedule else 0

    def on_epoch_end(self) -> None:
        schedule = self.training.schedule
        stages = len(schedule.history) if schedule else 0
        if (
            self.model.train_state.step % self.config.checkpoint.every_steps == 0
            or stages != self._stage_count
            or self.training.complete
        ):
            self.save()
        self._stage_count = stages

    def on_train_end(self) -> None:
        if self._last_saved_step != self.model.train_state.step:
            self.save()
