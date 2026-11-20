"""Сценарии обучения PINN для гауссова импульса.

Бэкенд обучения загружается только при вызове операции.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import PinnGaussianConfig

if TYPE_CHECKING:
    from lle.runs.runs import Run

__all__ = ["PinnGaussianConfig", "train", "resume", "extend", "transfer"]


def train(config: PinnGaussianConfig) -> Run:
    """Обучает новую модель и возвращает Run."""
    from .workflow import train as execute

    return execute(config)


def resume(source: Run | str) -> Run:
    """Возобновляет незавершённый Run по объекту Run или его ID."""
    from .workflow import resume as execute

    return execute(source)


def extend(source: Run | str, steps: int) -> Run:
    """Добавляет шаги обучения в новом дочернем Run."""
    from .workflow import extend as execute

    return execute(source, steps)


def transfer(source: Run | str, config: PinnGaussianConfig | None = None) -> Run:
    """Переносит веса в новый Run с новым состоянием оптимизатора."""
    from .workflow import transfer as execute

    return execute(source, config)
