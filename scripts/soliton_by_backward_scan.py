import numpy as np

from lle_solver.grid import PeriodicGrid
from lle_solver.integrators import rk4_step
from lle_solver.lle import LugiatoLefever
from lle_solver.split_schemes import SplitScheme, StrangSplitStep
from lle_solver.steps import FourierStep, IntegratorStep

from _common import count_peaks
from _viewer import FrameRecorder, write_viewer

SEED = 42

pump_amplitude = np.sqrt(8.7)
beta2 = -2.0

detuning_start = -2.9
detuning_captured = 8.0
scan_rate = 0.3
detuning_step = 0.05
capture_hold = 20.0

backward_step = 0.02
backward_hold = 2.0
max_backward_steps = 120
target_solitons = 1
confirm_hold = 50.0
settle_time = 100.0

num_points = 512
window = 30.0
time_step = 5e-4
noise_amplitude = 1e-3

frame_interval = 0.2

grid = PeriodicGrid(num_points, length=window)


def build_scheme(detuning: float) -> SplitScheme:
    equation = LugiatoLefever(detuning, pump_amplitude, beta2)
    return StrangSplitStep(
        FourierStep(equation.linear_operator, grid.wavenumbers),
        IntegratorStep(equation.nonlinear_term, rk4_step),
    )


recorder = FrameRecorder(grid, time_step, frame_interval)


def run_phase(
        field: np.ndarray, detuning: float, duration: float, phase: str, elapsed: int,
) -> tuple[np.ndarray, int]:
    scheme = build_scheme(detuning)
    num_steps = round(duration / time_step)
    for done in range(0, num_steps, recorder.steps_per_frame):
        chunk = min(recorder.steps_per_frame, num_steps - done)
        field = scheme.advance(field, chunk, time_step)
        recorder.record(field, elapsed + done + chunk, detuning, phase)
    return field, elapsed + num_steps


rng = np.random.RandomState(SEED)
field = noise_amplitude * np.exp(2j * np.pi * rng.rand(num_points))
elapsed = 0

for index in range(round((detuning_captured - detuning_start) / detuning_step)):
    field, elapsed = run_phase(
        field, detuning_start + detuning_step * (index + 0.5),
        detuning_step / scan_rate, "forward", elapsed,
    )
field, elapsed = run_phase(field, detuning_captured, capture_hold, "hold", elapsed)
captured_solitons = count_peaks(field)
print(f"захвачено солитонов: {captured_solitons}")

detuning = detuning_captured
achieved = False

for _ in range(max_backward_steps):
    detuning -= backward_step
    field, elapsed = run_phase(field, detuning, backward_hold, "backward", elapsed)
    num_solitons = count_peaks(field)
    print(f"  Δ={detuning:5.2f}  N={num_solitons:2d}  "
          f"<|F|^2>={np.mean(np.abs(field) ** 2):.3f}")

    if num_solitons == target_solitons:
        field, elapsed = run_phase(field, detuning, confirm_hold, "confirm", elapsed)
        achieved = count_peaks(field) == target_solitons
        if achieved:
            break
    elif num_solitons == 0 or num_solitons > captured_solitons:
        break

field, elapsed = run_phase(field, detuning, settle_time, "settle", elapsed)
num_solitons = count_peaks(field)

if achieved:
    print(f"получено N={target_solitons} при Δ={detuning:.2f}, "
          f"пиковая мощность {np.abs(field).max() ** 2:.2f}")
elif num_solitons == 0:
    print("все солитоны потеряны: уменьшите backward_step "
          "или увеличьте backward_hold")
elif num_solitons > captured_solitons:
    print(f"солитонное состояние развалилось при Δ={detuning:.2f}: "
          f"N={num_solitons} против {captured_solitons} захваченных, "
          f"фон потерял устойчивость")
else:
    print(f"остановка на N={num_solitons} при Δ={detuning:.2f}: "
          f"не хватило шагов, увеличьте max_backward_steps")

write_viewer(
    "soliton_by_backward_scan", recorder,
    title="Одиночный солитон обратным сканированием",
    subtitle=f"LLE, Strang split-step Fourier, прямой скан до Δ={detuning_captured}, "
             f"затем обратный проход шагом {backward_step}",
    phases={
        "forward": "прямой скан", "hold": f"выдержка при Δ={detuning_captured}",
        "backward": "обратный проход", "confirm": "подтверждение",
        "settle": "финальная выдержка",
    },
    pump_amplitude=pump_amplitude, beta2=beta2, seed=SEED,
    detuning_start=detuning_start, detuning_captured=detuning_captured,
    scan_rate=scan_rate, detuning_step=detuning_step,
    backward_step=backward_step, backward_hold=backward_hold,
    confirm_hold=confirm_hold, settle_time=settle_time,
    num_points=num_points, window=window, time_step=time_step,
    noise_amplitude=noise_amplitude,
    captured_solitons=captured_solitons,
    achieved=achieved, final_detuning=round(detuning, 4),
    final_solitons=num_solitons,
)