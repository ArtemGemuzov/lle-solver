import numpy as np

from lle.numerics.types import State, Symbol, Wavenumbers
from lle.physics.equation import LLEParameters


def linear_operator(equation: LLEParameters, wavenumbers: Wavenumbers) -> Symbol:
    return -1.0 - 1j * equation.detuning + 1j * equation.beta2 / 2.0 * wavenumbers**2


def nonlinear_term(equation: LLEParameters, state: State) -> State:
    return equation.pump_amplitude + 1j * np.abs(state) ** 2 * state
