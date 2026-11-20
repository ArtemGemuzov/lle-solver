"""Состояние обучения и восстановление внутренних объектов DeepXDE."""

from copy import deepcopy

import deepxde as dde
import torch

DATA_FIELDS = (
    "train_x_all",
    "train_x_bc",
    "num_bcs",
    "train_x",
    "train_y",
    "test_x",
    "test_y",
    "train_aux_vars",
    "test_aux_vars",
)


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value


class TrainingState(dde.callbacks.Callback):
    def __init__(self, resampler, schedule, iterations: int):
        super().__init__()
        self.resampler = resampler
        self.schedule = schedule
        self.iterations = iterations
        self._restored = False
        self._best_state = {}

    @property
    def complete(self) -> bool:
        return self.model.train_state.step >= self.iterations or (
            self.schedule is not None
            and len(self.schedule.history) == len(self.schedule.epsilons)
        )

    def state_dict(self) -> dict:
        model = self.model
        if not all(torch.isfinite(p).all().item() for p in model.net.parameters()):
            raise FloatingPointError("Cannot export non-finite PINN weights")
        return deepcopy(
            {
                "versions": {
                    "deepxde": dde.__version__,
                    "torch": str(torch.__version__),
                },
                "model": model.net.state_dict(),
                "optimizer": model.opt.state_dict(),
                "train_state": vars(model.train_state),
                "history": vars(model.losshistory),
                "loss_weights": model.loss_weights,
                "data": {key: getattr(model.data, key) for key in DATA_FIELDS},
                "weighting": model.data.state_dict()
                if hasattr(model.data, "state_dict")
                else None,
                "schedule": self.schedule.state_dict()
                if self.schedule is not None
                else None,
                "resample_counter": self.resampler.epochs_since_last_resample,
            }
        )

    def load_state_dict(self, checkpoint: dict) -> None:
        model = self.model
        device = next(model.net.parameters()).device
        model.net.load_state_dict(checkpoint["model"])
        model.opt.load_state_dict(checkpoint["optimizer"])
        vars(model.train_state).update(deepcopy(checkpoint["train_state"]))
        vars(model.losshistory).update(deepcopy(checkpoint["history"]))
        model.loss_weights = _to_device(checkpoint["loss_weights"], device)
        for key, value in deepcopy(checkpoint["data"]).items():
            setattr(model.data, key, value)
        if checkpoint["weighting"] is not None:
            model.data.load_state_dict(_to_device(checkpoint["weighting"], device))
        if self.schedule is not None:
            self.schedule.load_state_dict(deepcopy(checkpoint["schedule"]))
        self.resampler.epochs_since_last_resample = checkpoint["resample_counter"]
        self._best_state = {
            key: value
            for key, value in vars(model.train_state).items()
            if key.startswith("best_")
        }
        self._restored = True

    def on_train_begin(self):
        if self._restored:
            # DeepXDE вычисляет потери перед этим вызовом. Убираем повторную запись
            # в истории и восстанавливаем лучший результат из чекпоинта.
            for values in vars(self.model.losshistory).values():
                values.pop()
            vars(self.model.train_state).update(self._best_state)
            self._restored = False
