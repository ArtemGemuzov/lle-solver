"""Обучение PINN для эволюции гауссова импульса.

Запускайте из корня проекта в окружении с бэкендом PyTorch.

  python scripts/pinn_gaussian.py train --config experiments/05_pinn-single-soliton/pinn_config.yaml
  python scripts/pinn_gaussian.py train --config experiments/06_pinn-rad-sampling/rad_config.yaml
  python scripts/pinn_gaussian.py resume ИД
  python scripts/pinn_gaussian.py extend ИД --steps 80000
  python scripts/pinn_gaussian.py transfer ИД domain.time_end=4.5 training.iterations=20000

ИД — имя каталога запуска в runs/.
train обучает модель с нуля. Без --config используются параметры по умолчанию.
resume возобновляет незавершённый запуск с последнего чекпоинта.
extend создаёт дочерний запуск и добавляет шаги Adam с сохранением его состояния.
transfer переносит веса и начинает обучение с шага 0. По умолчанию используются
параметры родителя. Для другого базового конфига укажите --config.
В train и transfer отдельные параметры задаются в форме ключ=значение.
"""

import argparse
import sys
from pathlib import Path

from lle.config import load_config
from lle.runs.runs import get_run
from lle.workflows.pinn_gaussian.config import PinnGaussianConfig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Обучить модель с нуля")
    resume = commands.add_parser("resume", help="Возобновить прерванный запуск")
    extend = commands.add_parser("extend", help="Добавить шаги оптимизации")
    transfer = commands.add_parser("transfer", help="Перенести веса в новый запуск")

    for sub in (resume, extend, transfer):
        sub.add_argument("run_id", metavar="RUN_ID")
    for sub in (train, transfer):
        sub.add_argument("--config", type=Path, help="Базовый конфиг YAML")
        sub.add_argument("overrides", nargs="*", metavar="key=value")
    extend.add_argument(
        "--steps", required=True, type=int, help="Число дополнительных шагов Adam"
    )

    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in commands.choices:
        # Разрешаем параметры конфига как до, так и после --config.
        command, *options = argv
        args = commands.choices[command].parse_intermixed_args(options)
    else:
        parser.parse_args(argv)
        return
    if command == "extend" and args.steps < 1:
        extend.error("--steps must be positive")

    source = get_run(args.run_id) if command != "train" else None
    if command in ("train", "transfer"):
        path = args.config
        if command == "transfer" and path is None:
            path = source.path / "parameters.yaml"
        config = load_config(PinnGaussianConfig, path, args.overrides)

    from lle.workflows import pinn_gaussian

    if command == "train":
        pinn_gaussian.train_run(config)
    elif command == "resume":
        pinn_gaussian.resume(source)
    elif command == "extend":
        pinn_gaussian.extend(source, args.steps)
    else:
        pinn_gaussian.transfer(source, config)


if __name__ == "__main__":
    main()
