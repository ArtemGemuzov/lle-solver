"""Сценарии запуска PINN для гауссова импульса."""

import shutil
from pathlib import Path

import deepxde as dde
import torch

from lle.pinn.checkpoint import load_checkpoint
from lle.pinn.model import build_model, initial_weights
from lle.pinn.training import PinnTraining
from lle.runs.runs import PROJECT_ROOT, Run, create_run, get_run, resume_run
from lle.workflows.pinn_gaussian.config import PinnGaussianConfig

from .outputs import save_results, save_stage, training_runtime

# Сохраняем идентификатор сценария для совместимости с существующими Run.
DEFAULT_SCRIPT = PROJECT_ROOT / "scripts/pinn_gaussian.py"


def _source(source: Run | str) -> Run:
    return get_run(source) if isinstance(source, str) else source


def _training(run, config):
    dde.config.set_random_seed(config.training.seed)
    runtime = training_runtime(torch.get_default_device())
    metadata = run.load_metadata()
    metadata["runtime"] = runtime
    run.save_metadata(metadata)
    print(f"Запуск: {run.id}", flush=True)
    print(f"Устройство: {runtime['device']} ({runtime['device_name']})", flush=True)
    model = build_model(
        equation=config.equation,
        initial=config.initial,
        domain=config.domain,
        network=config.network,
        training=config.training,
        loss=config.loss,
    )
    return PinnTraining(
        model,
        config.training,
        checkpoint_path=run.path / "checkpoint.pt",
        checkpoint_every_steps=config.checkpoint.every_steps,
        parameters=config.model_dump(mode="json"),
        causal=config.loss.causal,
        on_stage_end=lambda stage: save_stage(run, config, model, stage),
    )


def _fit(run, config, training):
    history = training.fit()
    save_results(run, config, training, history)


def clear_resumed_outputs(path: Path, completed_stages: int) -> None:
    """Удаляет результаты, которые предстоит пересчитать после возобновления."""
    for name in (
        "losses.csv",
        "loss_weights.csv",
        "causal_stages.csv",
        "model.pt",
        "field_evolution.npz",
    ):
        (path / name).unlink(missing_ok=True)
    for stage in path.glob("stage_*"):
        suffix = stage.name.removeprefix("stage_")
        if stage.is_dir() and suffix.isdigit() and int(suffix) > completed_stages:
            shutil.rmtree(stage)


def train(config: PinnGaussianConfig, *, script=DEFAULT_SCRIPT) -> Run:
    """Обучает новую модель и возвращает завершённый Run."""
    with create_run(config, script=script) as run:
        _fit(run, config, _training(run, config))
    return run


def resume(source: Run | str, *, script=DEFAULT_SCRIPT) -> Run:
    """Продолжает незавершённый Run с последнего чекпоинта."""
    source = _source(source)
    with resume_run(source.path, script=script) as run:
        saved = load_checkpoint(run.path / "checkpoint.pt")
        config = PinnGaussianConfig.model_validate(saved["config"])
        if config != run.load_config(PinnGaussianConfig):
            raise ValueError(
                "Checkpoint configuration differs from the run configuration"
            )
        training = _training(run, config)
        training.checkpoint.restore(saved)
        schedule = training.checkpoint.schedule
        clear_resumed_outputs(run.path, len(schedule.history) if schedule else 0)
        _fit(run, config, training)
    return run


def extend(source: Run | str, steps: int, *, script=DEFAULT_SCRIPT) -> Run:
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
        training = _training(run, config)
        training.checkpoint.restore(saved)
        training.checkpoint.save()
        _fit(run, config, training)
    return run


def transfer(
    source: Run | str,
    config: PinnGaussianConfig | None = None,
    *,
    script=DEFAULT_SCRIPT,
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
    weights = initial_weights(
        parent["model"],
        time_ratio=config.domain.time_end / source_config.domain.time_end,
    )
    metadata = {
        "parent_run_id": source.id,
        "parent_step": parent["train_state"]["step"],
        "parent_mode": "init",
    }
    del parent
    with create_run(config, script=script, extra_metadata=metadata) as run:
        training = _training(run, config)
        training.model.net.load_state_dict(weights)
        _fit(run, config, training)
    return run
