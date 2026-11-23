"""Сценарии запуска PINN для гауссова импульса."""

import shutil
from functools import partial
from pathlib import Path

import deepxde as dde
import torch

from lle.pinn.features import PeriodicFeatures
from lle.pinn.model import build_model
from lle.pinn.training import PinnTraining, prepare_training, train_model
from lle.runs.checkpoint import load_checkpoint, restore_rng
from lle.runs.runs import PROJECT_ROOT, Run, create_run, get_run, resume_run
from lle.workflows.pinn_gaussian.config import PinnGaussianConfig

from .outputs import (
    save_results,
    save_stage,
    save_training_checkpoint,
    training_runtime,
)

# Сохраняем идентификатор сценария для совместимости с существующими Run.
DEFAULT_SCRIPT = PROJECT_ROOT / "scripts/pinn_gaussian.py"


def _source(source: Run | str) -> Run:
    return get_run(source) if isinstance(source, str) else source


def _build_model(config: PinnGaussianConfig) -> dde.Model:
    return build_model(
        config.to_problem(),
        network=config.network,
        loss=config.loss,
        seed=config.training.seed,
        interior_points=config.domain.interior_points,
        initial_points=config.domain.initial_points,
    )


def _model_for_run(run: Run, config: PinnGaussianConfig) -> dde.Model:
    runtime = training_runtime(torch.get_default_device())
    metadata = run.load_metadata()
    metadata["runtime"] = runtime
    run.save_metadata(metadata)
    print(f"Запуск: {run.id}", flush=True)
    print(f"Устройство: {runtime['device']} ({runtime['device_name']})", flush=True)
    return _build_model(config)


def _training_for_run(run: Run, config: PinnGaussianConfig) -> PinnTraining:
    model = _model_for_run(run, config)
    causal = config.loss.causal
    return prepare_training(
        model,
        config.training,
        rad=config.sampling.rad,
        schedule=causal.schedule if causal else None,
        on_stage_end=partial(save_stage, run, config, model),
    )


def _fit(run: Run, config: PinnGaussianConfig, training: PinnTraining) -> None:
    history = training.fit(
        on_checkpoint=partial(save_training_checkpoint, run, config),
        checkpoint_every=config.checkpoint.every_steps,
    )
    save_results(run, config, training.model, history)


def clear_resumed_outputs(path: Path, completed_stages: int) -> None:
    """Удаляет результаты, которые предстоит пересчитать после возобновления."""
    for name in (
        "losses.csv",
        "loss_weights.csv",
        "causal_stages.csv",
        "model.pt",
        "field_evolution.npz",
        "sampling.csv",
        "sampling_histograms.npz",
    ):
        (path / name).unlink(missing_ok=True)
    for stage in path.glob("stage_*"):
        suffix = stage.name.removeprefix("stage_")
        if stage.is_dir() and suffix.isdigit() and int(suffix) > completed_stages:
            shutil.rmtree(stage)


def train_run(
    config: PinnGaussianConfig, *, script: str | Path = DEFAULT_SCRIPT
) -> Run:
    """Обучает новую модель и возвращает завершённый Run."""
    with create_run(config, script=script) as run:
        model = _model_for_run(run, config)
        causal = config.loss.causal
        history = train_model(
            model,
            config.training,
            rad=config.sampling.rad,
            schedule=causal.schedule if causal else None,
            on_stage_end=partial(save_stage, run, config, model),
            on_checkpoint=partial(save_training_checkpoint, run, config),
            checkpoint_every=config.checkpoint.every_steps,
        )
        save_results(run, config, model, history)
    return run


def resume(source: Run | str) -> Run:
    """Продолжает незавершённый Run с последнего чекпоинта."""
    source = _source(source)
    with resume_run(source.path) as run:
        saved = load_checkpoint(run.path / "checkpoint.pt")
        config = PinnGaussianConfig.model_validate(saved["config"])
        if config != run.load_config(PinnGaussianConfig):
            raise ValueError(
                "Checkpoint configuration differs from the run configuration"
            )
        training = _training_for_run(run, config)
        training.load_state_dict(saved)
        restore_rng(saved)
        schedule = training.schedule
        clear_resumed_outputs(run.path, len(schedule.history) if schedule else 0)
        _fit(run, config, training)
    return run


def extend(
    source: Run | str, steps: int, *, script: str | Path = DEFAULT_SCRIPT
) -> Run:
    """Создаёт дочерний Run с дополнительными шагами и состоянием Adam."""
    source = _source(source)
    parent = load_checkpoint(source.path / "checkpoint.pt")
    source_config = PinnGaussianConfig.model_validate(parent["config"])
    if steps < 1:
        raise ValueError("The number of extra steps must be positive")
    causal = source_config.loss.causal
    if causal is not None and causal.schedule is not None:
        raise ValueError("Cannot extend training with loss.causal.schedule enabled")
    parameters = source_config.model_dump(mode="json")
    parameters["training"]["iterations"] = parent["train_state"]["step"] + steps
    config = PinnGaussianConfig.model_validate(parameters)
    saved = {**parent, "config": config.model_dump(mode="json")}
    metadata = {
        "parent_run_id": source.id,
        "parent_step": saved["train_state"]["step"],
        "parent_mode": "extend",
    }
    with create_run(config, script=script, extra_metadata=metadata) as run:
        training = _training_for_run(run, config)
        training.load_state_dict(saved)
        restore_rng(saved)
        save_training_checkpoint(run, config, training.state_dict())
        _fit(run, config, training)
    return run


def transfer(
    source: Run | str,
    config: PinnGaussianConfig | None = None,
    *,
    script: str | Path = DEFAULT_SCRIPT,
) -> Run:
    """Переносит веса в новый Run; оптимизатор и счётчики создаются заново."""
    source = _source(source)
    if config is None:
        config = source.load_config(PinnGaussianConfig)
    parent = load_checkpoint(source.path / "checkpoint.pt")
    source_config = PinnGaussianConfig.model_validate(parent["config"])
    if source_config.network != config.network:
        raise ValueError("Weight transfer requires matching network configurations")
    if source_config.domain.length != config.domain.length:
        raise ValueError(
            "Weight transfer requires matching domain.length for the spatial harmonics"
        )
    source_features = PeriodicFeatures(
        source_config.domain.length,
        source_config.domain.time_end,
        source_config.network.spatial_harmonics,
    )
    target_features = PeriodicFeatures(
        config.domain.length, config.domain.time_end, config.network.spatial_harmonics
    )
    weights = source_features.transfer_weights(parent["model"], target_features)
    metadata = {
        "parent_run_id": source.id,
        "parent_step": parent["train_state"]["step"],
        "parent_mode": "init",
    }
    del parent
    with create_run(config, script=script, extra_metadata=metadata) as run:
        training = _training_for_run(run, config)
        training.model.net.load_state_dict(weights)
        _fit(run, config, training)
    return run


train = train_run
