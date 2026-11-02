"""Эволюция гауссова импульса."""

from lle_runs.config import config_from_cli
from lle_runs.runs import create_run
import numpy as np

from lle_solver.simulations.gaussian import GaussianConfig
from lle_solver.simulations.evolution import FieldEvolution
from lle_solver.grid import PeriodicGrid
from lle_solver.integrators import rk4_step
from lle_solver.lle import LugiatoLefever
from lle_solver.split_schemes import StrangSplitStep
from lle_solver.steps import FourierStep, IntegratorStep


def simulate(p: GaussianConfig) -> FieldEvolution:
    dt = p.solver.time_step

    total = round(p.solver.time_end / dt)
    record_step = round(p.recording.interval / dt)

    grid = PeriodicGrid(p.grid.num_points, p.grid.length)
    equation = LugiatoLefever(
        p.equation.detuning,
        p.equation.pump_amplitude,
        p.equation.beta2,
    )
    scheme = StrangSplitStep(
        FourierStep(equation.linear_operator, grid.wavenumbers),
        IntegratorStep(equation.nonlinear_term, rk4_step),
    )

    field = (
        p.initial.amplitude
        * np.exp(-grid.theta**2 / (2 * p.initial.width**2))
    ).astype(np.complex128)

    steps = np.arange(0, total + 1, record_step)
    fields = np.empty((steps.size, grid.num_points), dtype=np.complex128)

    for index, step in enumerate(steps):
        if index:
            field = scheme.advance(field, record_step, dt)

        if not np.isfinite(field).all():
            raise FloatingPointError(f"Нечисловое поле на шаге {step}")

        fields[index] = field

    return FieldEvolution(
        theta=grid.theta,
        times=np.asarray(steps * dt, dtype=np.float64),
        fields=fields,
    )


def main() -> None:
    config = config_from_cli(GaussianConfig)
    with create_run(config, script=__file__) as run:
        simulate(config).save(run.path)
    print(f"Run: {run.id}")


if __name__ == "__main__":
    main()
