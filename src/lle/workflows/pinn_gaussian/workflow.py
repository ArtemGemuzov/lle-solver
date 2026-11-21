"""Сценарии запуска PINN для гауссова импульса."""

import shutil
from pathlib import Path

import deepxde as dde
import torch

from lle.physics.equation import LLEParameters
from lle.physics.initial import GaussianPulse
from lle.pinn.features import PeriodicFeatures
from lle.pinn.loss.causal import CausalTimePDE
from lle.pinn.loss.relobralo import ReLoBRaLoTimePDE
from lle.pinn.model import (
    build_network,
    lle_residual,
)
from lle.pinn.sampling import RADSampler
from lle.pinn.training import PinnTraining, configure_adam
from lle.runs.checkpoint import load_checkpoint, restore_rng, save_checkpoint
from lle.runs.runs import PROJECT_ROOT, Run, create_run, get_run, resume_run
from lle.workflows.pinn_gaussian.config import PinnGaussianConfig

from .outputs import TrainingCheckpoint, save_results, save_stage, training_runtime

# Сохраняем идентификатор сценария для совместимости с существующими Run.
DEFAULT_SCRIPT = PROJECT_ROOT / "scripts/pinn_gaussian.py"


def _source(source: Run | str) -> Run:
    return get_run(source) if isinstance(source, str) else source


def _build_model(config: PinnGaussianConfig) -> dde.Model:
    """Собирает обучающую задачу и выбирает взвешивание потерь."""
    domain = config.domain
    half_length = domain.length / 2
    geometry = dde.geometry.GeometryXTime(
        dde.geometry.Interval(-half_length, half_length),
        dde.geometry.TimeDomain(0, domain.time_end),
    )
    residual = lle_residual(LLEParameters(**config.equation.model_dump()))
    initial = GaussianPulse(**config.initial.model_dump())
    conditions = [
        dde.icbc.IC(
            geometry,
            lambda points: initial(points[:, :1]).real,
            lambda _, on: on,
            component=0,
        ),
        dde.icbc.IC(
            geometry,
            lambda points: initial(points[:, :1]).imag,
            lambda _, on: on,
            component=1,
        ),
    ]
    options = {
        "num_domain": domain.interior_points,
        "num_initial": domain.initial_points,
        "num_test": domain.interior_points,
        "train_distribution": "pseudo",
    }
    balancing = config.loss.relobralo
    causal = config.loss.causal
    if balancing is not None:
        data = ReLoBRaLoTimePDE(
            geometry,
            residual,
            conditions,
            **options,
            alpha=balancing.alpha,
            temperature=balancing.temperature,
            lookback_probability=balancing.lookback_probability,
            seed=config.training.seed,
        )
    elif causal is not None:
        data = CausalTimePDE(
            geometry,
            residual,
            conditions,
            **options,
            bands=causal.bands,
            epsilon=causal.schedule.epsilons[0] if causal.schedule else causal.epsilon,
            time_end=domain.time_end,
        )
    else:
        data = dde.data.TimePDE(geometry, residual, conditions, **options)
    # Сохраняем порядок генерации точек и инициализации весов для воспроизводимости.
    features = PeriodicFeatures(
        domain.length, domain.time_end, config.network.spatial_harmonics
    )
    net = build_network(
        input_size=features.output_size,
        hidden_layers=config.network.hidden_layers,
        hidden_width=config.network.hidden_width,
    )
    net.apply_feature_transform(features)
    return dde.Model(data, net)


def _training(run: Run, config: PinnGaussianConfig) -> PinnTraining:
    dde.config.set_random_seed(config.training.seed)
    runtime = training_runtime(torch.get_default_device())
    metadata = run.load_metadata()
    metadata["runtime"] = runtime
    run.save_metadata(metadata)
    print(f"Запуск: {run.id}", flush=True)
    print(f"Устройство: {runtime['device']} ({runtime['device_name']})", flush=True)
    model = _build_model(config)
    configure_adam(
        model,
        learning_rate=config.training.learning_rate,
        initial_weight=config.training.initial_weight,
    )
    rad = config.sampling.rad
    resampler = (
        RADSampler(
            period=config.training.resample_every,
            length=config.domain.length,
            time_end=config.domain.time_end,
            seed=config.training.seed,
            **rad.model_dump(),
        )
        if rad is not None
        else None
    )
    training = PinnTraining(
        model,
        iterations=config.training.iterations,
        resample_every=config.training.resample_every,
        log_every=config.training.log_every,
        resampler=resampler,
        causal_epsilons=config.loss.causal.schedule.epsilons
        if config.loss.causal and config.loss.causal.schedule
        else None,
        causal_threshold=config.loss.causal.schedule.threshold
        if config.loss.causal and config.loss.causal.schedule
        else 0.99,
        on_stage_end=lambda stage: save_stage(run, config, model, stage),
    )

    callback = TrainingCheckpoint(run, config, training)
    callback.set_model(model)
    training.callbacks.append(callback)
    return training


def _fit(run: Run, config: PinnGaussianConfig, training: PinnTraining) -> None:
    history = training.fit()
    save_checkpoint(
        run.path / "checkpoint.pt",
        training.state_dict(),
        config.model_dump(mode="json"),
    )
    save_results(run, config, training, history)


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


def train(config: PinnGaussianConfig, *, script: str | Path = DEFAULT_SCRIPT) -> Run:
    """Обучает новую модель и возвращает завершённый Run."""
    with create_run(config, script=script) as run:
        _fit(run, config, _training(run, config))
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
        training = _training(run, config)
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
        training = _training(run, config)
        training.load_state_dict(saved)
        restore_rng(saved)
        save_checkpoint(
            run.path / "checkpoint.pt",
            training.state_dict(),
            config.model_dump(mode="json"),
        )
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
        training = _training(run, config)
        training.model.net.load_state_dict(weights)
        _fit(run, config, training)
    return run
