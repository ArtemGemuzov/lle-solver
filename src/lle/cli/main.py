"""Запуск из корня проекта: lle ssfm solve solver.time_end=1.0"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from lle.config import load_config
from lle.runs.runs import get_run
from lle.workflows.pinn_gaussian.config import PinnGaussianConfig
from lle.workflows.ssfm_gaussian.config import SsfmGaussianConfig


def _config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="Базовый конфиг YAML")
    parser.add_argument("overrides", nargs="*", metavar="key=value")


def _execute(method: str, action: str, args: argparse.Namespace) -> None:
    if method == "ssfm":
        from lle.workflows.ssfm_gaussian.workflow import run

        result = run(load_config(SsfmGaussianConfig, args.config, args.overrides))
        print(f"Run: {result.id}")
        return

    from lle.workflows import pinn_gaussian

    if action == "train":
        pinn_gaussian.train_run(
            load_config(PinnGaussianConfig, args.config, args.overrides)
        )
    elif action == "resume":
        pinn_gaussian.resume(args.run_id)
    elif action == "extend":
        pinn_gaussian.extend(args.run_id, args.steps)
    else:
        source = get_run(args.run_id)
        path = args.config or source.path / "parameters.yaml"
        pinn_gaussian.transfer(
            source, load_config(PinnGaussianConfig, path, args.overrides)
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="lle", description="Численные расчеты LLE и обучение PINN"
    )
    methods = parser.add_subparsers(required=True)
    ssfm = methods.add_parser("ssfm", help="Спектральный решатель").add_subparsers(
        required=True
    )
    pinn = methods.add_parser("pinn", help="Обучение PINN").add_subparsers(
        required=True
    )
    solve = ssfm.add_parser("solve", help="Вычислить эволюцию поля")
    train = pinn.add_parser("train", help="Обучить модель с нуля")
    resume = pinn.add_parser("resume", help="Возобновить прерванный запуск")
    extend = pinn.add_parser("extend", help="Добавить шаги оптимизации")
    transfer = pinn.add_parser("transfer", help="Перенести веса в новый запуск")
    for command in (solve, train):
        command.description = "Начальное поле: гауссов импульс."
    for command in (resume, extend, transfer):
        command.add_argument("run_id", metavar="RUN_ID")
    for command in (solve, train, transfer):
        _config_args(command)
    extend.add_argument(
        "--steps", required=True, type=int, help="Число дополнительных шагов Adam"
    )

    tokens = list(sys.argv[1:] if argv is None else argv)
    actions = {"ssfm": ssfm, "pinn": pinn}
    if (
        len(tokens) < 2
        or tokens[0] not in actions
        or tokens[1] not in actions[tokens[0]].choices
    ):
        parser.parse_args(tokens)
        return
    method, action, *arguments = tokens
    command = actions[method].choices[action]
    # Переопределения разрешены по обе стороны от --config.
    args = command.parse_intermixed_args(arguments)
    if action == "extend" and args.steps < 1:
        command.error("--steps must be positive")
    try:
        _execute(method, action, args)
    except (ValueError, FileNotFoundError, yaml.YAMLError) as error:
        command.error(str(error))
