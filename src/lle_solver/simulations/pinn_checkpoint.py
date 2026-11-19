"""Полное состояние PINN и атомарная замена последнего чекпоинта."""

import os
from pathlib import Path
import random

import deepxde as dde
import numpy as np
import torch

from lle_solver.simulations.pinn_gaussian import PinnGaussianConfig


FORMAT_VERSION = 1
DATA_FIELDS = (
    "train_x_all", "train_x_bc", "num_bcs", "train_x", "train_y",
    "test_x", "test_y", "train_aux_vars", "test_aux_vars",
)


def _pack(value):
    if isinstance(value, np.ndarray):
        return {"__ndarray__": True, "dtype": value.dtype.str, "shape": value.shape,
                "bytes": torch.from_numpy(np.frombuffer(value.tobytes(), dtype=np.uint8).copy())}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _pack(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_pack(item) for item in value)
    if isinstance(value, list):
        return [_pack(item) for item in value]
    return value


def _unpack(value, device="cpu"):
    if isinstance(value, dict):
        if value.get("__ndarray__") is True:
            raw = value["bytes"].numpy().tobytes()
            return np.frombuffer(raw, dtype=value["dtype"]).copy().reshape(value["shape"])
        return {key: _unpack(item, device) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_unpack(item, device) for item in value)
    if isinstance(value, list):
        return [_unpack(item, device) for item in value]
    return value


def load_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != FORMAT_VERSION:
        raise ValueError("Неподдерживаемая версия чекпоинта PINN")
    PinnGaussianConfig.model_validate(checkpoint["config"])
    return checkpoint


def extend_checkpoint(checkpoint: dict, extra_steps: int) -> dict:
    """Prepare a child checkpoint with a new target and the same training state."""
    config = PinnGaussianConfig.model_validate(checkpoint["config"])
    if extra_steps < 1:
        raise ValueError("--extra-steps must be positive")
    if config.loss.causal is not None and config.loss.causal.schedule is not None:
        raise ValueError("--extend-from does not support loss.causal.schedule")
    parameters = config.model_dump(mode="json")
    parameters["training"]["iterations"] = checkpoint["train_state"]["step"] + extra_steps
    return {**checkpoint, "config": parameters}


class PinnCheckpoint(dde.callbacks.Callback):
    def __init__(self, path: Path, config: PinnGaussianConfig, resampler, schedule=None):
        super().__init__()
        self.path = path
        self.config = config
        self.resampler = resampler
        self.schedule = schedule
        self._restored = False
        self._best_state = {}
        self._stage_count = 0
        self._last_saved_step = None

    @property
    def complete(self):
        return (self.model.train_state.step >= self.config.training.iterations
                or (self.schedule is not None
                    and len(self.schedule.history) == len(self.schedule.config.epsilons)))

    def save(self):
        model = self.model
        if not all(torch.isfinite(p).all().item() for p in model.net.parameters()):
            raise FloatingPointError("Нечисловые веса PINN; предыдущий чекпоинт сохранён")
        checkpoint = {
            "format_version": FORMAT_VERSION,
            "config": self.config.model_dump(mode="json"),
            "versions": {"deepxde": dde.__version__, "torch": str(torch.__version__)},
            "model": model.net.state_dict(),
            "optimizer": model.opt.state_dict(),
            "train_state": vars(model.train_state),
            "history": vars(model.losshistory),
            "loss_weights": model.loss_weights,
            "data": {key: getattr(model.data, key) for key in DATA_FIELDS},
            "weighting": model.data.state_dict() if hasattr(model.data, "state_dict") else None,
            "schedule": self.schedule.state_dict() if self.schedule is not None else None,
            "resample_counter": self.resampler.epochs_since_last_resample,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
        }
        temporary = self.path.with_suffix(".tmp")
        try:
            with temporary.open("wb") as target:
                torch.save(_pack(checkpoint), target)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.path)
            self._last_saved_step = model.train_state.step
        finally:
            temporary.unlink(missing_ok=True)

    def restore(self, checkpoint):
        if checkpoint["config"] != self.config.model_dump(mode="json"):
            raise ValueError("Конфигурация не совпадает с чекпоинтом")
        model = self.model
        device = next(model.net.parameters()).device
        model.net.load_state_dict(checkpoint["model"])
        model.opt.load_state_dict(checkpoint["optimizer"])
        vars(model.train_state).update(_unpack(checkpoint["train_state"]))
        vars(model.losshistory).update(_unpack(checkpoint["history"]))
        model.loss_weights = _unpack(checkpoint["loss_weights"], device)
        for key, value in _unpack(checkpoint["data"]).items():
            setattr(model.data, key, value)
        if checkpoint["weighting"] is not None:
            model.data.load_state_dict(_unpack(checkpoint["weighting"], device))
        if self.schedule is not None:
            self.schedule.load_state_dict(_unpack(checkpoint["schedule"]))
            self._stage_count = len(self.schedule.history)
        self.resampler.epochs_since_last_resample = checkpoint["resample_counter"]
        # Building the model consumes randomness; restore generators only afterwards.
        rng = _unpack(checkpoint["rng"])
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        self._best_state = {key: value for key, value in vars(model.train_state).items()
                            if key.startswith("best_")}
        self._restored = True

    def on_train_begin(self):
        if self._restored:
            # DeepXDE evaluates once before this hook. It must not add an extra
            # history entry or change the saved best result on resampled points.
            for values in vars(self.model.losshistory).values():
                values.pop()
            vars(self.model.train_state).update(self._best_state)
            self._restored = False

    def on_epoch_end(self):
        stages = len(self.schedule.history) if self.schedule is not None else 0
        if (self.model.train_state.step % self.config.checkpoint.every_steps == 0
                or stages != self._stage_count or self.complete):
            self.save()
        self._stage_count = stages

    def on_train_end(self):
        if self._last_saved_step != self.model.train_state.step:
            self.save()
