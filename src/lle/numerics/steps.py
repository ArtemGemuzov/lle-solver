from functools import lru_cache
from typing import Protocol

import numpy as np
from numpy.fft import fft, ifft

from lle.numerics.types import (
    Integrator,
    LinearOperator,
    NonlinearTerm,
    State,
    StepSize,
    Symbol,
    Wavenumbers,
)


class Step(Protocol):
    def __call__(self, state: State, step_size: StepSize) -> State: ...


class FourierStep:
    def __init__(
        self, linear_operator: LinearOperator, wavenumbers: Wavenumbers
    ) -> None:
        self.symbol = linear_operator(wavenumbers)
        self._propagator = lru_cache(maxsize=2)(self._compute_propagator)

    def _compute_propagator(self, step_size: StepSize) -> Symbol:
        return np.exp(step_size * self.symbol)

    def __call__(self, state: State, step_size: StepSize) -> State:
        spectrum = fft(state)
        return ifft(self._propagator(step_size) * spectrum)


class IntegratorStep:
    def __init__(self, nonlinear_term: NonlinearTerm, integrator: Integrator) -> None:
        self.nonlinear_term = nonlinear_term
        self.integrator = integrator

    def __call__(self, state: State, step_size: StepSize) -> State:
        return self.integrator(self.nonlinear_term, state, step_size)
