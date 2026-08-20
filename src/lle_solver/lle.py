from lle_solver.types import Wavenumbers, Symbol, State

import numpy as np


class LugiatoLefever:
    def __init__(self, detuning: float, pump_amplitude: float, beta2: float) -> None:
        self.detuning = detuning
        self.pump_amplitude = pump_amplitude
        self.beta2 = beta2

    def linear_operator(self, wavenumbers: Wavenumbers) -> Symbol:
        return -1.0 - 1j * self.detuning + 1j * self.beta2 / 2.0 * wavenumbers ** 2

    def nonlinear_term(self, state: State) -> State:
        return self.pump_amplitude + 1j * np.abs(state) ** 2 * state