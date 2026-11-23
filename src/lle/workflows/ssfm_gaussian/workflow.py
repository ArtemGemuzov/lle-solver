from lle.numerics.ssfm import solve
from lle.physics.fields import FieldEvolution
from lle.runs.runs import PROJECT_ROOT, Run, create_run
from lle.storage import save_evolution
from lle.workflows.ssfm_gaussian.config import SsfmGaussianConfig


def simulate(p: SsfmGaussianConfig) -> FieldEvolution:
    """Вычисляет эволюцию гауссова импульса."""
    return solve(
        p.to_problem(),
        num_points=p.grid.num_points,
        time_step=p.solver.time_step,
        recording_interval=p.recording.interval,
    )


def run(config: SsfmGaussianConfig) -> Run:
    """Вычисляет эволюцию, сохраняет поле и возвращает завершенный запуск."""
    with create_run(config, script=PROJECT_ROOT / "scripts/gaussian.py") as result:
        save_evolution(result.path, simulate(config))
    return result
