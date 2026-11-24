"""Сценарии запуска PINN для гауссова импульса."""

import shutil
from functools import partial
from pathlib import Path

import torch

from lle.pinn import solver
from lle.pinn.features import PeriodicFeatures
from lle.pinn.model import build_periodic_network
from lle.pinn.parameters import RADParameters
from lle.runs.checkpoint import load_checkpoint
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


def _record_runtime(run: Run) -> None:
    runtime = training_runtime(torch.get_default_device())
    metadata = run.load_metadata()
    metadata["runtime"] = runtime
    run.save_metadata(metadata)
    print(f"Запуск: {run.id}", flush=True)
    print(f"Устройство: {runtime['device']} ({runtime['device_name']})", flush=True)


def _inputs(config: PinnGaussianConfig):
    problem = config.to_problem()
    features = PeriodicFeatures(
        problem.length, problem.time_end, config.network.spatial_harmonics
    )
    torch.manual_seed(config.training.seed)
    network = build_periodic_network(
        features,
        hidden_layers=config.network.hidden_layers,
        hidden_width=config.network.hidden_width,
    )
    return network, problem.equation, problem.domain, problem.initial


def _keywords(run: Run, config: PinnGaussianConfig) -> solver.TrainingKeywords:
    _record_runtime(run)
    training = config.training
    return {
        "interior_points": config.domain.interior_points,
        "initial_points": config.domain.initial_points,
        "learning_rate": training.learning_rate,
        "initial_weight": training.initial_weight,
        "resample_every": training.resample_every,
        "log_every": training.log_every,
        "seed": training.seed,
        "loss": config.loss_parameters(),
        "rad": RADParameters(**config.sampling.rad.model_dump())
        if config.sampling.rad
        else None,
        "on_stage_end": partial(save_stage, run, config),
        "on_checkpoint": partial(save_training_checkpoint, run, config),
        "checkpoint_every": config.checkpoint.every_steps,
    }


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
        solution = solver.train(
            *_inputs(config), steps=config.training.iterations, **_keywords(run, config)
        )
        save_results(run, config, solution.model, solution.history)
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
        schedule = saved["schedule"]
        clear_resumed_outputs(run.path, len(schedule["history"]) if schedule else 0)
        solution = solver.resume(
            *_inputs(config),
            state=saved,
            steps=config.training.iterations,
            **_keywords(run, config),
        )
        save_results(run, config, solution.model, solution.history)
    return run


def extend(
    source: Run | str, steps: int, *, script: str | Path = DEFAULT_SCRIPT
) -> Run:
    """Создаёт дочерний Run с дополнительными шагами и состоянием Adam."""
    source = _source(source)
    parent = load_checkpoint(source.path / "checkpoint.pt")
    source_config = PinnGaussianConfig.model_validate(parent["config"])
    budget = solver.extension_budget(parent, steps, source_config.loss_parameters())
    parameters = source_config.model_dump(mode="json")
    parameters["training"]["iterations"] = budget
    config = PinnGaussianConfig.model_validate(parameters)
    metadata = {
        "parent_run_id": source.id,
        "parent_step": parent["train_state"]["step"],
        "parent_mode": "extend",
    }
    with create_run(config, script=script, extra_metadata=metadata) as run:
        save_training_checkpoint(run, config, parent)
        solution = solver.extend(
            *_inputs(config), state=parent, steps=steps, **_keywords(run, config)
        )
        save_results(run, config, solution.model, solution.history)
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
        raise ValueError("Weight transfer requires matching network architectures")
    if source_config.domain.length != config.domain.length:
        raise ValueError("Weight transfer requires matching domain.length")
    features = PeriodicFeatures(
        source_config.domain.length,
        source_config.domain.time_end,
        source_config.network.spatial_harmonics,
    )
    metadata = {
        "parent_run_id": source.id,
        "parent_step": parent["train_state"]["step"],
        "parent_mode": "init",
    }
    with create_run(config, script=script, extra_metadata=metadata) as run:
        solution = solver.transfer(
            *_inputs(config),
            state=parent,
            steps=config.training.iterations,
            source_features=features,
            **_keywords(run, config),
        )
        save_results(run, config, solution.model, solution.history)
    return run


train = train_run
