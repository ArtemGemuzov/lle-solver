from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TypedDict, Unpack

import deepxde as dde
import torch
from deepxde.model import LossHistory
from numpy.typing import ArrayLike

from lle.physics.equation import LLEParameters
from lle.physics.fields import FieldEvolution
from lle.physics.problem import InitialField, LLEProblem, RingDomain
from lle.pinn.features import PeriodicFeatures
from lle.pinn.loss.schedule import CausalStage
from lle.pinn.model import build_model
from lle.pinn.parameters import LossParameters, RADParameters, TrainingParameters
from lle.pinn.training import PinnTraining, predict_evolution, prepare_training


@dataclass(frozen=True)
class PinnSolution:
    model: dde.Model
    history: LossHistory
    state: dict

    def predict(self, *, theta: ArrayLike, times: ArrayLike) -> FieldEvolution:
        """Вычисляет комплексное поле обученной модели на заданных осях."""
        return predict_evolution(self.model.net, theta=theta, times=times)


class TrainingEvents(TypedDict, total=False):
    on_stage_end: Callable[[dde.Model, CausalStage], None]
    on_checkpoint: Callable[[dict], None]
    checkpoint_every: int


class TrainingKeywords(TrainingEvents, total=False):
    interior_points: int
    initial_points: int
    learning_rate: float
    initial_weight: float
    resample_every: int
    log_every: int
    seed: int
    loss: LossParameters
    rad: RADParameters | None


def _prepare(
    network: torch.nn.Module,
    equation: LLEParameters,
    domain: RingDomain,
    initial: InitialField,
    steps: int,
    *,
    interior_points: int = 1024,
    initial_points: int = 256,
    learning_rate: float = 0.001,
    initial_weight: float = 10.0,
    resample_every: int = 100,
    log_every: int = 100,
    seed: int = 42,
    loss: LossParameters | None = None,
    rad: RADParameters | None = None,
    **events: Unpack[TrainingEvents],
) -> PinnTraining:
    if loss is None:
        loss = LossParameters()
    training = TrainingParameters(
        iterations=steps,
        learning_rate=learning_rate,
        initial_weight=initial_weight,
        resample_every=resample_every,
        log_every=log_every,
        seed=seed,
    )
    problem = LLEProblem(equation, initial, domain.length, domain.time_end)
    model = build_model(
        problem,
        network=network,
        loss=loss,
        seed=seed,
        interior_points=interior_points,
        initial_points=initial_points,
    )
    callback = events.get("on_stage_end")
    return prepare_training(
        model,
        training,
        rad=rad,
        schedule=loss.causal.schedule if loss.causal else None,
        on_stage_end=partial(callback, model) if callback is not None else None,
    )


def _fit(session: PinnTraining, events: TrainingEvents) -> PinnSolution:
    final_state = None
    callback = events.get("on_checkpoint")

    def report(state: dict) -> None:
        nonlocal final_state
        if session.complete:
            final_state = state
        if callback is not None:
            callback(state)

    history = session.fit(
        on_checkpoint=report if callback is not None else None,
        checkpoint_every=events.get("checkpoint_every", 1000),
    )
    if final_state is None:
        final_state = session.state_dict()
    return PinnSolution(session.model, history, final_state)


def train(
    network: torch.nn.Module,
    equation: LLEParameters,
    domain: RingDomain,
    initial: InitialField,
    *,
    steps: int,
    **kwargs: Unpack[TrainingKeywords],
) -> PinnSolution:
    """Обучает переданную сеть на LLE с заданным начальным полем."""
    return _fit(_prepare(network, equation, domain, initial, steps, **kwargs), kwargs)


def resume(
    network: torch.nn.Module,
    equation: LLEParameters,
    domain: RingDomain,
    initial: InitialField,
    state: dict,
    *,
    steps: int,
    **kwargs: Unpack[TrainingKeywords],
) -> PinnSolution:
    """Восстанавливает снимок той же задачи и обучает до общего числа шагов steps."""
    if state["train_state"]["step"] > steps:
        raise ValueError("Checkpoint step exceeds the training budget")
    if "rng" not in state:
        raise ValueError("Resuming training requires a snapshot with RNG state")
    session = _prepare(network, equation, domain, initial, steps, **kwargs)
    session.load_state_dict(state)
    if session.complete:
        final = session.state_dict()
        if callback := kwargs.get("on_checkpoint"):
            callback(final)
        return PinnSolution(session.model, session.model.losshistory, final)
    return _fit(session, kwargs)


def extension_budget(state: dict, steps: int, loss: LossParameters) -> int:
    if type(steps) is not int or steps < 1:
        raise ValueError("The number of extra steps must be positive")
    if loss.causal and loss.causal.schedule:
        raise ValueError("Cannot extend training with a causal schedule")
    return state["train_state"]["step"] + steps


def extend(
    network: torch.nn.Module,
    equation: LLEParameters,
    domain: RingDomain,
    initial: InitialField,
    state: dict,
    *,
    steps: int,
    **kwargs: Unpack[TrainingKeywords],
) -> PinnSolution:
    """Продолжает обучение со снимка еще на steps шагов Adam."""
    budget = extension_budget(state, steps, kwargs.get("loss", LossParameters()))
    return resume(network, equation, domain, initial, state, steps=budget, **kwargs)


def transfer(
    network: torch.nn.Module,
    equation: LLEParameters,
    domain: RingDomain,
    initial: InitialField,
    state: dict,
    *,
    steps: int,
    source_features: PeriodicFeatures | None = None,
    **kwargs: Unpack[TrainingKeywords],
) -> PinnSolution:
    """Переносит веса в целевую сеть и обучает ее с новым Adam."""
    weights = state["model"]
    if source_features is not None:
        target = getattr(network, "_input_transform", None)
        if not isinstance(target, PeriodicFeatures):
            raise ValueError(
                "Periodic weight transfer requires target periodic features"
            )
        weights = source_features.transfer_weights(weights, target)
    network.load_state_dict(weights)
    return train(network, equation, domain, initial, steps=steps, **kwargs)
