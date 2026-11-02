import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict
import yaml


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True,
                              allow_inf_nan=False, validate_default=True)


C = TypeVar("C", bound=Config)


def load_config(model: type[C], path: Path | None = None,
                overrides: Sequence[str] = ()) -> C:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(values, dict):
        raise ValueError("Конфигурация должна быть YAML-словарём")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"Ожидается имя=значение: {override}")
        target = values
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Не является вложенной конфигурацией: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return model.model_validate(values)


def config_from_cli(model: type[C]) -> C:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("overrides", nargs="*", metavar="key=value")
    args = parser.parse_args()
    try:
        return load_config(model, args.config, args.overrides)
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
