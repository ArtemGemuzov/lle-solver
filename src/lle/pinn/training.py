"""Обучение модели и предсказание поля без зависимости от Run."""

from collections.abc import Callable, Sequence

import deepxde as dde
import numpy as np
import torch
from deepxde.model import LossHistory

from lle.numerics.grid import PeriodicGrid
from lle.physics.fields import FieldEvolution
from lle.pinn.loss.causal import CausalTimePDE
from lle.pinn.loss.relobralo import ReLoBRaLoTimePDE
from lle.pinn.loss.schedule import CausalSchedule, CausalStage
from lle.pinn.model import predict_field
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
    length: float,
    time_end: float,
    interval: float,
    num_points: int,
) -> FieldEvolution:
    grid = PeriodicGrid(num_points, length)
    count = round(time_end / interval)
    times = np.arange(count + 1, dtype=np.float64) * interval
    theta, time = np.meshgrid(grid.theta, times)
    points = np.column_stack((theta.ravel(), time.ravel()))
    values = predict_field(net, points).reshape(theta.shape)
    if not np.isfinite(values).all():
        raise FloatingPointError("PINN predicted non-finite field values")
    return FieldEvolution(theta=grid.theta, times=times, values=values)


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
    ) -> None:
        if model.opt is None:
            raise ValueError("Configure the optimizer before creating PinnTraining")
        self.model = model
        self.iterations = iterations
        self.log_every = log_every
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

    def fit(self) -> LossHistory:
        model = self.model
        schedule = self.schedule
        remaining = 0 if self.complete else self.iterations - model.train_state.step
        history, state = model.train(
            iterations=remaining,
            display_every=self.log_every,
            callbacks=self.callbacks,
        )
        if schedule is not None and history.steps[-1] != state.step:
            history, _ = model.train(iterations=0, verbose=0)
        return history
