"""Сравнивает четыре горизонта обучения базовой PINN."""

import sys

from _run_pinn_t25_sweep import Experiment, run_experiments


EXPERIMENTS = tuple(
    Experiment(
        name=f"T={time_end:.1f}",
        layers=4,
        width=64,
        harmonics=4,
        points=1024,
        steps=160_000,
        time_end=time_end,
    )
    for time_end in (2.1, 2.2, 2.3, 2.4)
)


if __name__ == "__main__":
    sys.exit(run_experiments(EXPERIMENTS))
