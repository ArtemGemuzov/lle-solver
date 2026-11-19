"""Сценарии запуска PINN для гауссова импульса."""

from pathlib import Path
import shutil

from lle_runs.runs import Run, create_run, resume_run
from lle_solver.simulations.pinn_checkpoint import extend_checkpoint, initial_weights, load_checkpoint
from lle_solver.simulations.pinn_gaussian import PinnGaussianConfig
from lle_solver.simulations.pinn_training import PinnTraining


def clear_resumed_outputs(path: Path, completed_stages: int) -> None:
    """Удаляет результаты, которые предстоит пересчитать после возобновления."""
    for name in ("losses.csv", "loss_weights.csv", "causal_stages.csv",
                 "model.pt", "field_evolution.npz"):
        (path / name).unlink(missing_ok=True)
    for stage in path.glob("stage_*"):
        suffix = stage.name.removeprefix("stage_")
        if stage.is_dir() and suffix.isdigit() and int(suffix) > completed_stages:
            shutil.rmtree(stage)


def train(script, config):
    with create_run(config, script=script) as run:
        PinnTraining(run, config).fit()


def resume(script, source: Run):
    with resume_run(source.path, script=script) as run:
        saved = load_checkpoint(run.path / "checkpoint.pt")
        config = PinnGaussianConfig.model_validate(saved["config"])
        if config != run.load_config(PinnGaussianConfig):
            raise ValueError("Checkpoint configuration differs from the run configuration")
        training = PinnTraining(run, config)
        training.checkpoint.restore(saved)
        schedule = training.checkpoint.schedule
        clear_resumed_outputs(run.path, len(schedule.history) if schedule else 0)
        training.fit()


def extend(script, source: Run, steps: int):
    parent = load_checkpoint(source.path / "checkpoint.pt")
    saved = extend_checkpoint(parent, steps)
    config = PinnGaussianConfig.model_validate(saved["config"])
    metadata = {"parent_run_id": source.id, "parent_step": saved["train_state"]["step"],
                "parent_mode": "extend"}
    with create_run(config, script=script, extra_metadata=metadata) as run:
        training = PinnTraining(run, config)
        training.checkpoint.restore(saved)
        training.checkpoint.save()
        training.fit()


def transfer(script, source: Run, config: PinnGaussianConfig):
    parent = load_checkpoint(source.path / "checkpoint.pt")
    weights = initial_weights(parent, config)
    metadata = {"parent_run_id": source.id, "parent_step": parent["train_state"]["step"],
                "parent_mode": "init"}
    del parent
    with create_run(config, script=script, extra_metadata=metadata) as run:
        training = PinnTraining(run, config)
        training.model.net.load_state_dict(weights)
        training.fit()
