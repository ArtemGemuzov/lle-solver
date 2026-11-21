"""Сравнивает равномерную выборку и RAD на T = 4 за 500000 шагов Adam.

Запуск из корня репозитория на GPU-сервере:
  conda activate a.gemuzov
  python scripts/_run_pinn_sweep.py --dry-run
  nohup python -u scripts/_run_pinn_sweep.py >> pinn_sweep.log 2>&1 < /dev/null &

Каждое обучение начинается после выхода предыдущего процесса.
"""

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER = Path("scripts/pinn_gaussian.py")
CONFIG = Path("experiments/06_pinn-rad-sampling/rad_config.yaml")
TIME_END = 4.0
ITERATIONS = 500_000


@dataclass(frozen=True)
class Experiment:
    points: int
    resample_every: int
    use_rad: bool

    @property
    def name(self) -> str:
        sampler = "RAD" if self.use_rad else "uniform"
        return f"{sampler}, N={self.points}, resample_every={self.resample_every}"

    def command(self) -> list[str]:
        command = [
            sys.executable,
            str(TRAINER),
            "train",
            "--config",
            str(CONFIG),
            f"domain.time_end={TIME_END}",
            f"domain.interior_points={self.points}",
            f"training.iterations={ITERATIONS}",
            f"training.resample_every={self.resample_every}",
        ]
        if not self.use_rad:
            command.append("sampling.rad=null")
        return command


EXPERIMENTS = tuple(
    Experiment(points, period, use_rad)
    for points in (32_768, 16_384, 8_192)
    for period in (1_000, 100)
    for use_rad in (False, True)
)


def run_experiments(experiments: tuple[Experiment, ...]) -> int:
    env = dict(os.environ, DDE_BACKEND="pytorch", PYTHONUNBUFFERED="1")
    failed = []
    try:
        for experiment in experiments:
            name = experiment.name
            print(f"[{name}] Запуск", flush=True)
            result = subprocess.run(
                experiment.command(),
                cwd=PROJECT_ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                print(f"[{name}] Завершён", flush=True)
            else:
                failed.append(name)
                print(f"[{name}] Ошибка (код {result.returncode})", flush=True)
    except KeyboardInterrupt:
        print("Запуск серии прерван", flush=True)
        return 130

    print(f"Успешно завершено {len(experiments) - len(failed)} из {len(experiments)}")
    return int(bool(failed))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="Показать план запусков")
    args = parser.parse_args()

    print(
        f"T={TIME_END:g}, steps={ITERATIONS}, запусков: {len(EXPERIMENTS)}", flush=True
    )
    if args.dry_run:
        for experiment in EXPERIMENTS:
            print(experiment.name)
        return 0
    return run_experiments(EXPERIMENTS)


if __name__ == "__main__":
    sys.exit(main())
