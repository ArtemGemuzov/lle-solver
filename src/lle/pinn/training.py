"""Обучение модели и предсказание поля без зависимости от Run."""

import deepxde as dde
import numpy as np

from lle.fields import FieldEvolution
from lle.numerics.grid import PeriodicGrid
from lle.pinn.checkpoint import PinnCheckpoint
from lle.pinn.loss import CausalSchedule


def predict_evolution(
    model, *, length: float, time_end: float, interval: float, num_points: int
) -> FieldEvolution:
    grid = PeriodicGrid(num_points, length)
    count = round(time_end / interval)
    times = np.arange(count + 1, dtype=np.float64) * interval
    theta, time = np.meshgrid(grid.theta, times)
    points = np.column_stack((theta.ravel(), time.ravel()))
    prediction = np.asarray(model.predict(points), dtype=np.float64)
    values = (prediction[:, 0] + 1j * prediction[:, 1]).reshape(theta.shape)
    if not np.isfinite(values).all():
        raise FloatingPointError("PINN predicted non-finite field values")
    return FieldEvolution(theta=grid.theta, times=times, values=values)


class PinnTraining:
    """Модель, параметры обучения и состояние чекпоинта."""

    def __init__(
        self,
        model,
        training,
        *,
        checkpoint_path,
        checkpoint_every_steps: int,
        parameters,
        causal=None,
        on_stage_end=None,
    ):
        self.model = model
        self.training = training
        resampler = dde.callbacks.PDEPointResampler(period=training.resample_every)
        callbacks: list[dde.callbacks.Callback] = [resampler]
        schedule = None
        if causal is not None and causal.schedule is not None:
            stage_limit = training.iterations // len(causal.schedule.epsilons)
            schedule = CausalSchedule(
                causal.schedule, stage_limit, on_stage_end or (lambda stage: None)
            )
            callbacks.append(schedule)
        checkpoint = PinnCheckpoint(
            checkpoint_path,
            parameters,
            resampler,
            schedule,
            iterations=training.iterations,
            every_steps=checkpoint_every_steps,
        )
        checkpoint.set_model(model)
        self.checkpoint = checkpoint
        self.callbacks = [*callbacks, checkpoint]

    def fit(self):
        model = self.model
        checkpoint = self.checkpoint
        training = self.training
        schedule = checkpoint.schedule
        remaining = (
            0 if checkpoint.complete else training.iterations - model.train_state.step
        )
        history, state = model.train(
            iterations=remaining,
            display_every=training.log_every,
            callbacks=self.callbacks,
        )
        if schedule is not None and history.steps[-1] != state.step:
            history, _ = model.train(iterations=0, verbose=0)
            checkpoint.save()
        return history
