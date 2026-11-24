from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        allow_inf_nan=False,
        validate_default=True,
    )


C = TypeVar("C", bound=Config)


def load_config(
    model: type[C], path: Path | None = None, overrides: Sequence[str] = ()
) -> C:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) if path else {}
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a YAML mapping")
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator or not key or any(not part for part in key.split(".")):
            raise ValueError(f"Expected a key=value override: {override}")
        target = values
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Expected a nested configuration for key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return model.model_validate(values)
