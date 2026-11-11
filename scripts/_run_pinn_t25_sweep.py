"""Запускает девять экспериментов PINN на [0, 2.5]."""

from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from time import sleep


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINER = Path("scripts/pinn_gaussian.py")
CONFIG = Path("experiments/05_pinn-single-soliton/pinn_config.yaml")
MAX_PROCESSES = 4


@dataclass(frozen=True)
class Experiment:
    name: str
    layers: int
    width: int
    harmonics: int
    points: int
    steps: int
    time_end: float = 2.5

    def command(self) -> list[str]:
        return [
            sys.executable, str(TRAINER), "--config", str(CONFIG),
            f"domain.time_end={self.time_end}",
            f"domain.interior_points={self.points}",
            f"network.hidden_layers={self.layers}",
            f"network.hidden_width={self.width}",
            f"network.spatial_harmonics={self.harmonics}",
            f"training.iterations={self.steps}",
            "recording.interval=0.01",
        ]


EXPERIMENTS = (
    Experiment("05_combined", layers=4, width=128, harmonics=12, points=4096, steps=320_000),
    Experiment("01_more_steps", layers=4, width=64, harmonics=4, points=1024, steps=320_000),
    Experiment("02_more_points", layers=4, width=64, harmonics=4, points=4096, steps=160_000),
    Experiment("03_more_harmonics", layers=4, width=64, harmonics=12, points=1024, steps=160_000),
    Experiment("04_wider", layers=4, width=128, harmonics=4, points=1024, steps=160_000),
    Experiment("06_combined_more_steps", layers=4, width=128, harmonics=12, points=4096, steps=640_000),
    Experiment("07_combined_more_points", layers=4, width=128, harmonics=12, points=8192, steps=320_000),
    Experiment("08_combined_deeper", layers=6, width=128, harmonics=12, points=4096, steps=320_000),
    Experiment("09_combined_wider", layers=4, width=256, harmonics=12, points=4096, steps=320_000),
)


def run_experiments(experiments: tuple[Experiment, ...]) -> int:
    env = dict(os.environ, DDE_BACKEND="pytorch")
    pending = deque(experiments)
    active: dict[subprocess.Popen, str] = {}
    failed = []
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

            while pending and len(active) < MAX_PROCESSES:
                experiment = pending.popleft()
                process = subprocess.Popen(
                    experiment.command(), cwd=PROJECT_ROOT, env=env,
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                active[process] = experiment.name
                print(f"[{experiment.name}] Запущен (PID {process.pid})", flush=True)

            if active:
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
