"""Проверяет число точек и шагов Adam для базовой PINN до T = 3.5 и 4."""

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER = Path("scripts/pinn_gaussian.py")
CONFIG = Path("experiments/05_pinn-single-soliton/pinn_config.yaml")
MAX_PROCESSES = 4
START_INTERVAL_SECONDS = 10


@dataclass(frozen=True)
class Experiment:
    time_end: float
    points: int
    iterations: int

    @property
    def name(self) -> str:
        return f"T={self.time_end:g}, N={self.points}, steps={self.iterations}"

    def command(self) -> list[str]:
        return [
            sys.executable, str(TRAINER), "--config", str(CONFIG),
            f"domain.time_end={self.time_end}",
            f"domain.interior_points={self.points}",
            f"training.iterations={self.iterations}",
            "recording.interval=0.01",
        ]


EXPERIMENTS = (
    Experiment(time_end=3.5, points=16_384, iterations=320_000),
    Experiment(time_end=3.5, points=32_768, iterations=160_000),
    Experiment(time_end=4.0, points=32_768, iterations=320_000),
    Experiment(time_end=4.0, points=65_536, iterations=160_000),
)


def run_experiments(experiments: tuple[Experiment, ...]) -> int:
    env = dict(os.environ, DDE_BACKEND="pytorch")
    pending = deque(experiments)
    active: dict[subprocess.Popen, str] = {}
    failed = []
    next_start = monotonic()
    try:
        while pending or active:
            for process, name in list(active.items()):
                exit_code = process.poll()
                if exit_code is None:
                    continue
                del active[process]
                if exit_code == 0:
                    print(f"[{name}] Завершён", flush=True)
                else:
                    failed.append(name)
                    print(f"[{name}] Ошибка (код {exit_code})", flush=True)

            # Разносим старты, чтобы процессы не получили одинаковые временные ID.
            if pending and len(active) < MAX_PROCESSES and monotonic() >= next_start:
                experiment = pending.popleft()
                process = subprocess.Popen(
                    experiment.command(), cwd=PROJECT_ROOT, env=env,
                    stdin=subprocess.DEVNULL,
                )
                active[process] = experiment.name
                next_start = monotonic() + START_INTERVAL_SECONDS
                print(f"[{experiment.name}] Запущен (PID {process.pid})", flush=True)

            if pending or active:
                sleep(0.25)
    except KeyboardInterrupt:
        print("Запуск серии прерван", flush=True)
        return 130
    finally:
        for process in active:
            process.terminate()
        for process in active:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    print(f"Успешно завершено {len(experiments) - len(failed)} из {len(experiments)}")
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(run_experiments(EXPERIMENTS))
