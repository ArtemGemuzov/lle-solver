"""Эволюция гауссова импульса.

Из корня проекта: python scripts/gaussian.py solver.time_end=1.0 recording.interval=0.1
Без --config используются параметры по умолчанию.
"""

from lle.config import config_from_cli
from lle.runs.runs import create_run
from lle.storage import save_evolution
from lle.workflows.ssfm_gaussian import SsfmGaussianConfig, simulate


def main() -> None:
    config = config_from_cli(SsfmGaussianConfig)
    with create_run(config, script=__file__) as run:
        save_evolution(run.path, simulate(config))
    print(f"Run: {run.id}")


if __name__ == "__main__":
    main()
