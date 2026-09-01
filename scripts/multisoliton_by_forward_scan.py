import numpy as np

from lle_solver.evolution import evolve
from lle_solver.grid import PeriodicGrid
from lle_solver.integrators import rk4_step
from lle_solver.lle import LugiatoLefever
from lle_solver.split_schemes import SplitScheme, StrangSplitStep
from lle_solver.steps import FourierStep, IntegratorStep

from _viewer import FrameRecorder, write_viewer

SEED = 42

pump_amplitude = np.sqrt(8.7)
beta2 = -2.0

detuning_start = -2.9
detuning_end = 8.0
scan_rate = 0.01
detuning_step = 0.01

num_points = 1024
window = 60.0
time_step = 5e-4
noise_amplitude = 1e-3
check_time = 100.0

frame_interval = 0.3

grid = PeriodicGrid(num_points, length=window)


def build_scheme(detuning: float) -> SplitScheme:
    equation = LugiatoLefever(detuning, pump_amplitude, beta2)
    return StrangSplitStep(
        FourierStep(equation.linear_operator, grid.wavenumbers),
        IntegratorStep(equation.nonlinear_term, rk4_step),
    )


num_segments = round((detuning_end - detuning_start) / detuning_step)
steps_per_segment = round(detuning_step / scan_rate / time_step)

# Расстройка берётся в середине сегмента: это точное среднее по сегменту,
# иначе лестница смещена относительно скана на половину шага
segments = [
    (detuning_start + detuning_step * (index + 0.5), steps_per_segment, "forward")
    for index in range(num_segments)
]
segments.append((detuning_end, round(check_time / time_step), "check"))

rng = np.random.RandomState(SEED)
field = noise_amplitude * np.exp(2j * np.pi * rng.rand(num_points))

recorder = FrameRecorder(grid, frame_interval)
elapsed = 0.0
peak_power = []

for detuning, num_steps, phase in segments:
    scheme = build_scheme(detuning)
    for step, time, field in evolve(scheme, field, num_steps, time_step):
        if step == 0:
            continue
        recorder.record(field, elapsed + time, detuning, phase)
        if phase == "check":
            peak_power.append(np.abs(field).max() ** 2)
    elapsed += num_steps * time_step

peak_power = np.array(peak_power)
variation = (peak_power.max() - peak_power.min()) / peak_power.mean()
print(f"разброс пиковой мощности при Δ={detuning_end}: {variation * 100:.2f}%")

write_viewer(
    "multisoliton_by_forward_scan", recorder,
    title="Захват солитонов сканированием расстройки",
    subtitle=f"LLE, Strang split-step Fourier, Δ: {detuning_start} -> {detuning_end} "
             f"со скоростью {scan_rate}",
    phases={"forward": "скан расстройки", "check": f"выдержка при Δ={detuning_end}"},
    pump_amplitude=pump_amplitude, beta2=beta2, seed=SEED,
    detuning_start=detuning_start, detuning_end=detuning_end,
    scan_rate=scan_rate, detuning_step=detuning_step,
    num_points=num_points, window=window, time_step=time_step,
    noise_amplitude=noise_amplitude, check_time=check_time,
    peak_power_variation=round(float(variation), 5),
)