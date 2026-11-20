"""Стадии и расписание causal weighting."""

from collections.abc import Callable, Sequence
from typing import NamedTuple

import deepxde as dde


class CausalStage(NamedTuple):
    number: int
    epsilon: float
    first_step: int
    last_step: int
    min_weight: float
    reason: str


class CausalSchedule(dde.callbacks.Callback):
    def __init__(
        self,
        epsilons: Sequence[float],
        threshold: float,
        stage_limit: int,
        on_stage_end: Callable[[CausalStage], None],
    ):
        super().__init__()
        self.epsilons = tuple(epsilons)
        self.threshold = threshold
        self.stage_limit = stage_limit
        self.on_stage_end = on_stage_end
        self.history: list[CausalStage] = []
        self.first_step = 1

    def state_dict(self):
        return {
            "first_step": self.first_step,
            "history": [tuple(s) for s in self.history],
        }

    def load_state_dict(self, state):
        self.first_step = state["first_step"]
        self.history = [CausalStage(*stage) for stage in state["history"]]

    def on_batch_end(self):
        step = self.model.train_state.step
        min_weight = self.model.data.last_min_weight
        reached_threshold = min_weight > self.threshold
        if not reached_threshold and step - self.first_step + 1 < self.stage_limit:
            return

        stage = CausalStage(
            len(self.history) + 1,
            self.model.data.epsilon,
            self.first_step,
            step,
            min_weight,
            "threshold" if reached_threshold else "step_limit",
        )
        self.history.append(stage)
        self.on_stage_end(stage)
        if stage.number == len(self.epsilons):
            self.model.stop_training = True
        else:
            self.first_step = step + 1
            self.model.data.epsilon = self.epsilons[stage.number]
