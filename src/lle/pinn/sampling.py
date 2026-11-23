"""Перераспределение внутренних точек по невязке уравнения."""

from copy import deepcopy
from time import perf_counter

import deepxde as dde
import numpy as np

from lle.pinn.model import evaluate_residual


def residual_probabilities(residual, power: float, uniform_fraction: float):
    values = np.asarray(residual, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite residual in RAD candidates")
    if values.ndim != 1 or not len(values) or np.any(values < 0):
        raise ValueError("Expected a nonempty array of nonnegative residuals")
    # Масштабирование предотвращает переполнение при возведении в степень.
    maximum = values.max()
    if maximum == 0:
        return np.full(len(values), 1 / len(values))
    weights = (values / maximum) ** power
    return uniform_fraction / len(values) + (1 - uniform_fraction) * (
        weights / weights.sum()
    )


class RADSampler(dde.callbacks.PDEPointResampler):
    """Обновляет PDE-точки, сохраняя IC-выборку и тестовую сетку."""

    def __init__(
        self,
        *,
        period: int,
        length: float,
        time_end: float,
        candidate_points: int,
        batch_size: int,
        power: float,
        uniform_fraction: float,
        warmup_steps: int,
        seed: int,
        replace: bool = True,
    ):
        super().__init__(period=period)
        self.length = length
        self.time_end = time_end
        self.candidate_points = candidate_points
        self.replace = replace
        self.batch_size = batch_size
        self.power = power
        self.uniform_fraction = uniform_fraction
        self.warmup_steps = warmup_steps
        self.rng = np.random.default_rng(seed)
        self.last_update = None
        self.history = []

    def state_dict(self):
        return deepcopy(
            {
                "kind": "rad",
                "counter": self.epochs_since_last_resample,
                "rng": self.rng.bit_generator.state,
                "last_update": self.last_update,
                "history": self.history,
            }
        )

    def load_state_dict(self, state):
        self.epochs_since_last_resample = state["counter"]
        self.rng.bit_generator.state = deepcopy(state["rng"])
        self.last_update = state["last_update"]
        self.history = deepcopy(state["history"])

    def on_train_begin(self):
        super().on_train_begin()
        if self.warmup_steps == 0 and self.last_update is None:
            self.update_points()

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step < self.warmup_steps:
            super().on_epoch_end()
        elif (step - self.warmup_steps) % self.period == 0:
            self.update_points()

    def update_points(self):
        started = perf_counter()
        model = self.model
        data = model.data
        points = self.rng.uniform(
            [-self.length / 2, 0],
            [self.length / 2, self.time_end],
            size=(self.candidate_points, 2),
        ).astype(dde.config.real(np))
        residuals = []
        was_training = model.net.training
        try:
            for start in range(0, len(points), self.batch_size):
                components = evaluate_residual(
                    model.net, points[start : start + self.batch_size], data.pde
                )
                real, imag = [v.ravel().astype(np.float64) for v in components]
                residuals.append(np.hypot(real, imag))
        finally:
            model.net.train(was_training)
        residual = np.concatenate(residuals)
        probabilities = residual_probabilities(
            residual, self.power, self.uniform_fraction
        )
        indices = self.rng.choice(
            len(points), size=data.num_domain, replace=self.replace, p=probabilities
        )
        selected = points[indices]
        # PDE на начальном срезе сохраняет равномерную выборку, как у TimePDE.
        initial = np.column_stack(
            (
                self.rng.uniform(-self.length / 2, self.length / 2, data.num_initial),
                np.zeros(data.num_initial),
            )
        )
        data.replace_with_anchors(np.vstack((initial, selected)))
        self.last_update = model.train_state.step
        self.epochs_since_last_resample = 0
        histogram, _, _ = np.histogram2d(
            selected[:, 0],
            selected[:, 1],
            bins=16,
            range=[[-self.length / 2, self.length / 2], [0, self.time_end]],
        )
        self.history.append(
            {
                "step": self.last_update,
                "residual_mean": float(residual.mean()),
                "residual_max": float(residual.max()),
                "residual_p90": float(np.quantile(residual, 0.9)),
                "unique_points": len(np.unique(indices)),
                "seconds": perf_counter() - started,
                "histogram": histogram.astype(np.int64),
            }
        )
