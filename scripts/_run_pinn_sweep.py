"""Сравнивает равномерную выборку и RAD с пулами 4N и 16N на T = 5.

Запуск из корня репозитория на GPU-сервере:
  conda activate a.gemuzov
  python scripts/_run_pinn_sweep.py --dry-run
  nohup python -u scripts/_run_pinn_sweep.py >> pinn_t5_pool_sweep.log 2>&1 < /dev/null &

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
TIME_END = 5.0
ITERATIONS = 2_000_000


@dataclass(frozen=True)
class Experiment:
    points: int
    candidate_points: int | None

    @property
    def name(self) -> str:
        if self.candidate_points is None:
            return f"Uniform, N={self.points}"
        return f"RAD, N={self.points}, candidate_points={self.candidate_points}"

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
            "training.resample_every=100",
        ]
        if self.candidate_points is None:
            command.append("sampling.rad=null")
        else:
            command.append(f"sampling.rad.candidate_points={self.candidate_points}")
        return command


EXPERIMENTS = tuple(
    Experiment(points, candidates)
    for points in (131_072, 32_768)
    for candidates in (None, 4 * points, 16 * points)
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
