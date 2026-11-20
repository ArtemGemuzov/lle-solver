from functools import partial

import numpy as np
from _common import plot_maps, run

from lle.numerics.grid import PeriodicGrid
from lle.numerics.integrators import rk2_step
from lle.numerics.lle import linear_operator, nonlinear_term
from lle.numerics.split_schemes import LieTrotterSplitStep
from lle.numerics.steps import FourierStep, IntegratorStep
from lle.physics.equation import LLEParameters

SEED = 42

detuning = 0.0
pump_amplitude = np.sqrt(3)
beta2 = -2.0 / 25

num_points = 512
time_step = 1e-3
time_end = 100.0

grid = PeriodicGrid(num_points)
equation = LLEParameters(detuning, pump_amplitude, beta2)
scheme = LieTrotterSplitStep(
    FourierStep(partial(linear_operator, equation), grid.wavenumbers),
    IntegratorStep(partial(nonlinear_term, equation), rk2_step),
)

# Однородный фон плюс малый шум, который «затравливает» модуляционную неустойчивость
init_intensity = 1.0
noise_level = 0.02
rng = np.random.RandomState(SEED)
field = np.sqrt(init_intensity) * np.ones(num_points) + noise_level * rng.randn(
    num_points
)
field = field.astype(np.complex128)

field, history = run(
    scheme,
    grid,
    field,
    time_step,
    round(time_end / time_step),
    title="turing,",
)

np.savez(
    "turing.npz",
    field=field,
    times=history.times,
    energy=history.energy,
    detuning=detuning,
    pump_amplitude=pump_amplitude,
    beta2=beta2,
    num_points=num_points,
    time_step=time_step,
    seed=SEED,
)
plot_maps(grid, history)
