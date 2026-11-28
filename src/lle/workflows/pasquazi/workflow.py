from pathlib import Path

import numpy as np

from lle.numerics.grid import PeriodicGrid
from lle.numerics.pasquazi import solve
from lle.physics.initial import GaussianPulse
from lle.physics.pasquazi import PasquaziEvolution, PasquaziParameters
from lle.runs.runs import Run, create_run
from lle.storage import save_pasquazi
from lle.workflows.pasquazi.config import PasquaziConfig


def simulate(config: PasquaziConfig) -> PasquaziEvolution:
    """Запускает гауссов импульс в кольце при нулевых начальных полях петли."""
    grid = PeriodicGrid(config.num_points, 1.0)
    state = np.zeros(
        (2 * config.equation.n + 2, config.num_points), dtype=np.complex128
    )
    state[0] = GaussianPulse(**config.initial.model_dump())(grid.theta)
    return solve(
        PasquaziParameters(**config.equation.model_dump()),
        grid,
        state,
        time_step=config.solver.time_step,
        time_end=config.solver.time_end,
        recording_interval=config.recording.interval,
    )


def run(config: PasquaziConfig) -> Run:
    with create_run(config, script=Path(__file__)) as result:
        save_pasquazi(result.path, simulate(config))
    return result
