"""Обучение модели и предсказание поля без зависимости от Run."""

from collections.abc import Callable, Sequence

import deepxde as dde
import numpy as np
import torch
from deepxde.model import LossHistory
from numpy.typing import ArrayLike

from lle.physics.fields import FieldEvolution
from lle.pinn.loss.causal import CausalTimePDE
from lle.pinn.loss.relobralo import ReLoBRaLoTimePDE
from lle.pinn.loss.schedule import CausalSchedule, CausalStage
from lle.pinn.model import predict_field
from lle.pinn.parameters import CausalScheduleConfig, PinnTrainingConfig, RADConfig
from lle.pinn.sampling import RADSampler
from lle.pinn.state import TrainingState


def configure_adam(
    model: dde.Model, *, learning_rate: float, initial_weight: float
) -> None:
    """Создаёт Adam и подключает обновление адаптивных весов потерь."""
    if model.opt is not None:
        raise ValueError("The model already has an optimizer")
    weight = 1.0 if isinstance(model.data, ReLoBRaLoTimePDE) else initial_weight
    model.compile("adam", lr=learning_rate, loss_weights=[1.0, 1.0, weight, weight])
    if isinstance(model.data, (ReLoBRaLoTimePDE, CausalTimePDE)):
        model.opt.register_step_pre_hook(model.data.before_optimizer_step)


def predict_evolution(
    net: torch.nn.Module,
    *,
    theta: ArrayLike,
    times: ArrayLike,
) -> FieldEvolution:
    """Вычисляет поле на декартовом произведении заданных осей."""
    coordinates = np.asarray(theta, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    for axis in (coordinates, times):
        if axis.ndim != 1 or not axis.size or not np.isfinite(axis).all():
            raise ValueError("Prediction axes must be nonempty finite vectors")
    values = np.empty((times.size, coordinates.size), dtype=np.complex128)
    flat = values.ravel()
    batch_size = 8192
    for start in range(0, flat.size, batch_size):
        indices = np.arange(start, min(start + batch_size, flat.size))
        rows, columns = np.divmod(indices, coordinates.size)
        points = np.column_stack((coordinates[columns], times[rows]))
        flat[start : start + len(indices)] = predict_field(net, points)
    if not np.isfinite(values).all():
        raise FloatingPointError("PINN predicted non-finite field values")
    return FieldEvolution(theta=coordinates, times=times, values=values)


class PinnTraining:
    """Модель, параметры обучения и состояние чекпоинта."""

    def __init__(
        self,
        model: dde.Model,
        *,
        iterations: int,
        resample_every: int,
        log_every: int,
        causal_epsilons: Sequence[float] | None = None,
        causal_threshold: float = 0.99,
        on_stage_end: Callable[[CausalStage], None] | None = None,
        resampler: dde.callbacks.PDEPointResampler | None = None,
    ) -> None:
        if model.opt is None:
            raise ValueError("Configure the optimizer before creating PinnTraining")
        self.model = model
        self.iterations = iterations
        self.log_every = log_every
        if resampler is None:
            resampler = dde.callbacks.PDEPointResampler(period=resample_every)
        callbacks: list[dde.callbacks.Callback] = [resampler]
        schedule = None
        if causal_epsilons is not None:
            stage_limit = iterations // len(causal_epsilons)
            schedule = CausalSchedule(
                causal_epsilons,
                causal_threshold,
                stage_limit,
                on_stage_end or (lambda stage: None),
            )
            callbacks.append(schedule)
        self.state = TrainingState(resampler, schedule, iterations)
        self.state.set_model(model)
        self.callbacks = [*callbacks, self.state]

    @property
    def complete(self) -> bool:
        return self.state.complete

    @property
    def schedule(self) -> CausalSchedule | None:
        return self.state.schedule

    def state_dict(self) -> dict:
        return self.state.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.state.load_state_dict(state)

    def fit(
        self,
        *,
        on_checkpoint: Callable[[dict], None] | None = None,
        checkpoint_every: int = 1000,
    ) -> LossHistory:
        checkpoint = None
        callbacks = list(self.callbacks)
        if on_checkpoint is not None:
            if checkpoint_every < 1:
                raise ValueError("Checkpoint interval must be positive")
            checkpoint = TrainingCheckpoint(self, checkpoint_every, on_checkpoint)
            callbacks.append(checkpoint)
        model = self.model
        schedule = self.schedule
        remaining = 0 if self.complete else self.iterations - model.train_state.step
        history, state = model.train(
            iterations=remaining,
            display_every=self.log_every,
            callbacks=callbacks,
        )
        if schedule is not None and history.steps[-1] != state.step:
            history, _ = model.train(iterations=0, verbose=0)
        if checkpoint is not None:
            checkpoint.save()
        return history


class TrainingCheckpoint(dde.callbacks.Callback):
    """Передает снимок состояния после обновления точек и завершения стадий."""

    def __init__(
        self,
        training: PinnTraining,
        every_steps: int,
        on_checkpoint: Callable[[dict], None],
    ):
        super().__init__()
        self.every_steps = every_steps
        self.on_checkpoint = on_checkpoint
        self.training = training
        self._stage_count = 0

    def save(self) -> None:
        self.on_checkpoint(self.training.state_dict())

    def on_train_begin(self) -> None:
        schedule = self.training.schedule
        self._stage_count = len(schedule.history) if schedule else 0

    def on_epoch_end(self) -> None:
        schedule = self.training.schedule
        stages = len(schedule.history) if schedule else 0
        if not self.training.complete and (
            self.model.train_state.step % self.every_steps == 0
            or stages != self._stage_count
        ):
            self.save()
        self._stage_count = stages


def prepare_training(
    model: dde.Model,
    training: PinnTrainingConfig,
    *,
    rad: RADConfig | None = None,
    schedule: CausalScheduleConfig | None = None,
    on_stage_end: Callable[[CausalStage], None] | None = None,
) -> PinnTraining:
    """Готовит Adam, выборку и состояние для обучения или восстановления."""
    if schedule is not None and (
        not isinstance(model.data, CausalTimePDE)
        or training.iterations < len(schedule.epsilons)
        or model.data.epsilon != schedule.epsilons[0]
    ):
        raise ValueError("Causal schedule must match the model and iteration budget")
    resampler = None
    if rad is not None:
        if isinstance(model.data, (ReLoBRaLoTimePDE, CausalTimePDE)):
            raise ValueError("RAD currently requires fixed loss weights")
        if rad.candidate_points < model.data.num_domain:
            raise ValueError("RAD candidate_points must be at least interior_points")
        geometry = model.data.geom
        resampler = RADSampler(
            period=training.resample_every,
            length=float(geometry.geometry.r - geometry.geometry.l),
            time_end=float(geometry.timedomain.t1),
            seed=training.seed,
            **rad.model_dump(),
        )
    configure_adam(
        model,
        learning_rate=training.learning_rate,
        initial_weight=training.initial_weight,
    )
    return PinnTraining(
        model,
        iterations=training.iterations,
        resample_every=training.resample_every,
        log_every=training.log_every,
        resampler=resampler,
        causal_epsilons=schedule.epsilons if schedule else None,
        causal_threshold=schedule.threshold if schedule else 0.99,
        on_stage_end=on_stage_end,
    )


def train_model(
    model: dde.Model,
    training: PinnTrainingConfig,
    *,
    rad: RADConfig | None = None,
    schedule: CausalScheduleConfig | None = None,
    on_stage_end: Callable[[CausalStage], None] | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
    checkpoint_every: int = 1000,
) -> LossHistory:
    """Обучает переданную модель и возвращает историю потерь."""
    session = prepare_training(
        model, training, rad=rad, schedule=schedule, on_stage_end=on_stage_end
    )
    return session.fit(on_checkpoint=on_checkpoint, checkpoint_every=checkpoint_every)
