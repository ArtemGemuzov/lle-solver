"""Эволюция гауссова импульса.

Из корня проекта: python scripts/gaussian.py solver.time_end=1.0 recording.interval=0.1
Без --config используются параметры по умолчанию.
"""

from lle.config import config_from_cli
from lle.workflows.ssfm_gaussian.config import SsfmGaussianConfig
from lle.workflows.ssfm_gaussian.workflow import run


def main() -> None:
    config = config_from_cli(SsfmGaussianConfig)
    result = run(config)
    print(f"Run: {result.id}")


if __name__ == "__main__":
    main()
