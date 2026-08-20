from collections.abc import Iterator
from typing import TypeAlias

from lle_solver.types import State, StepSize, Time
from lle_solver.split_schemes import SplitScheme


Frame: TypeAlias = tuple[int, Time, State]


def evolve(
        scheme: SplitScheme,
        state: State,
        num_steps: int,
        step_size: StepSize,
) -> Iterator[Frame]:
    yield 0, 0.0, state
    for index in range(1, num_steps + 1):
        state = scheme.step(state, step_size)
        yield index, index * step_size, state