from typing import Protocol

from lle_solver.types import State, StepSize
from lle_solver.steps import Step


class SplitScheme(Protocol):
    def step(
        self,
        state: State,
        step_size: StepSize,
    ) -> State: ...


class LieTrotterSplitStep:
    def __init__(self, linear_step: Step, nonlinear_step: Step) -> None:
        self.linear_step = linear_step
        self.nonlinear_step = nonlinear_step

    def step(self, state: State, step_size: StepSize) -> State:
        state = self.linear_step(state, step_size)
        state = self.nonlinear_step(state, step_size)
        return state


class StrangSplitStep:
    def __init__(self, linear_step: Step, nonlinear_step: Step) -> None:
        self.linear_step = linear_step
        self.nonlinear_step = nonlinear_step

    def step(self, state: State, step_size: StepSize) -> State:
        state = self.linear_step(state, step_size / 2.0)
        state = self.nonlinear_step(state, step_size)
        state = self.linear_step(state, step_size / 2.0)
        return state