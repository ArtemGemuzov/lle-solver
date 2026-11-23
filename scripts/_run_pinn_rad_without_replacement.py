"""Проверяет RAD без возвращения на T = 4 за 500000 шагов Adam.

Запуск из корня репозитория на GPU-сервере:
  conda activate a.gemuzov
  python scripts/_run_pinn_rad_without_replacement.py --dry-run
  nohup python -u scripts/_run_pinn_rad_without_replacement.py >> pinn_t4_no_replace.log 2>&1 < /dev/null &
"""

import argparse

from _run_pinn_sweep import Experiment, run_experiments

EXPERIMENTS = tuple(
    Experiment(points, candidates, time_end=4.0, iterations=500_000, replace=False)
    for points, candidates in (
        (32_768, 131_072),
        (32_768, 524_288),
        (8_192, 131_072),
        (16_384, 131_072),
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="Показать план запусков")
    args = parser.parse_args()

    print(f"RAD без возвращения, запусков: {len(EXPERIMENTS)}", flush=True)
    if args.dry_run:
        for experiment in EXPERIMENTS:
            print(experiment.name)
        return 0
    return run_experiments(EXPERIMENTS)


if __name__ == "__main__":
    raise SystemExit(main())
