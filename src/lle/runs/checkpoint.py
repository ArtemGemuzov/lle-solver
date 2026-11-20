"""Атомарное хранение состояния и глобальных генераторов случайных чисел."""

import os
import random
from pathlib import Path

import numpy as np
import torch

FORMAT_VERSION = 1


def _pack(value):
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": value.dtype.str,
            "shape": value.shape,
            "bytes": torch.from_numpy(
                np.frombuffer(value.tobytes(), dtype=np.uint8).copy()
            ),
        }
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
            return (
                np.frombuffer(raw, dtype=value["dtype"]).copy().reshape(value["shape"])
            )
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
        raise ValueError("Unsupported checkpoint version")
    return _unpack(checkpoint)


def restore_rng(checkpoint: dict) -> None:
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def save_checkpoint(path: Path, state: dict, config: dict) -> None:
    checkpoint = {
        **state,
        "format_version": FORMAT_VERSION,
        "config": config,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as target:
            torch.save(_pack(checkpoint), target)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
