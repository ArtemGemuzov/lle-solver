from collections.abc import Callable
from typing import TypeAlias

import numpy as np

Time: TypeAlias = float
State: TypeAlias = np.ndarray
Spectrum: TypeAlias = np.ndarray
Wavenumbers: TypeAlias = np.ndarray
Symbol: TypeAlias = np.ndarray # Значения линейного оператора для всех фурье-мод
StepSize: TypeAlias = float

LinearOperator: TypeAlias = Callable[[Wavenumbers], Symbol]
NonlinearTerm: TypeAlias = Callable[[State], State]

# Все решаемые уравнения автономны
Derivative: TypeAlias = Callable[[State], State]
Integrator: TypeAlias = Callable[[Derivative, State, StepSize], State]