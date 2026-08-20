import numpy as np

from lle_solver.grid import PeriodicGrid
from lle_solver.integrators import rk2_step
from lle_solver.lle import LugiatoLefever
from lle_solver.split_schemes import LieTrotterSplitStep
from lle_solver.steps import FourierStep, IntegratorStep

from _common import plot_maps, run

detuning = 2.0
pump_amplitude = 1.37225
d2 = -0.002
beta2 = 2.0 * d2          # в нашей форме -i*beta2/2 * u_thetatheta == -i*d2 * u_thetatheta

num_points = 512
time_step = 1e-3
time_end = 30.0

grid = PeriodicGrid(num_points, length=2)
equation = LugiatoLefever(detuning, pump_amplitude, beta2)
scheme = LieTrotterSplitStep(
    FourierStep(equation.linear_operator, grid.wavenumbers),
    IntegratorStep(equation.nonlinear_term, rk2_step),
)

gaussian_pulse = (0.5 + np.exp(-(grid.theta / 0.85) ** 2)).astype(np.complex128)
field, history = run(
    scheme, grid, gaussian_pulse, time_step,
    round(time_end / time_step), title="soliton molecule,",
)

np.savez("soliton_molecule.npz", field=field, times=history.times,
         detuning=detuning, pump=pump_amplitude, beta2=beta2,
         num_points=num_points, time_step=time_step)
plot_maps(grid, history)