import numpy as np
from scipy.signal import find_peaks

from lle_solver.evolution import evolve
from lle_solver.grid import PeriodicGrid
from lle_solver.integrators import rk4_step
from lle_solver.lle import LugiatoLefever
from lle_solver.split_schemes import StrangSplitStep
from lle_solver.steps import FourierStep, IntegratorStep

from _viewer import FrameRecorder, write_viewer

SEED = 42

pump_amplitude = np.sqrt(8.7)
detuning = 4.5
beta2 = -2.0

num_points = 2048
window = 40.0
time_step = 5e-4
time_end = 8.0

frame_interval = 0.02
transient_time = 1.0

grid = PeriodicGrid(num_points, length=window)
equation = LugiatoLefever(detuning, pump_amplitude, beta2)
scheme = StrangSplitStep(
    FourierStep(equation.linear_operator, grid.wavenumbers),
    IntegratorStep(equation.nonlinear_term, rk4_step),
)

# Приближённое аналитическое решение для солитона
field = np.sqrt(2 * detuning) / np.cosh(np.sqrt(detuning) * grid.theta)
field = field.astype(np.complex128)

recorder = FrameRecorder(grid, frame_interval)
num_steps = round(time_end / time_step)
peak_power = np.empty(num_steps + 1)

for step, time, field in evolve(scheme, field, num_steps, time_step):
    peak_power[step] = np.abs(field).max() ** 2
    recorder.record(field, time, detuning, "evolution")

maxima, _ = find_peaks(peak_power[round(transient_time / time_step):])
period = np.mean(np.diff(maxima)) * time_step if maxima.size > 2 else np.nan
print(f"период дыхания: {period:.3f}")

write_viewer(
    "breather", recorder,
    title="Дышащий диссипативный солитон",
    subtitle=f"LLE, Strang split-step Fourier, фиксированная Δ={detuning}",
    phases={"evolution": f"эволюция при Δ={detuning}"},
    pump_amplitude=pump_amplitude, detuning=detuning, beta2=beta2,
    num_points=num_points, window=window, time_step=time_step, time_end=time_end,
    breathing_period=round(float(period), 4),
)