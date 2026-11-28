from pathlib import Path

import numpy as np

from lle.physics.fields import FieldEvolution
from lle.physics.pasquazi import PasquaziEvolution

_EVOLUTION_FILENAME = "field_evolution.npz"


def load_evolution(run_dir: Path) -> FieldEvolution:
    with np.load(run_dir / _EVOLUTION_FILENAME, allow_pickle=False) as data:
        return FieldEvolution(
            theta=data["theta"], times=data["times"], values=data["fields"]
        )


def save_evolution(run_dir: Path, evolution: FieldEvolution) -> None:
    with (run_dir / _EVOLUTION_FILENAME).open("xb") as target:
        np.savez_compressed(
            target,
            theta=evolution.theta,
            times=evolution.times,
            fields=evolution.values,
        )


def save_pasquazi(run_dir: Path, evolution: PasquaziEvolution) -> None:
    with (run_dir / _EVOLUTION_FILENAME).open("xb") as target:
        np.savez_compressed(
            target,
            theta=evolution.ring.theta,
            times=evolution.ring.times,
            fields=evolution.ring.values,
            loop=evolution.loop,
        )


def load_pasquazi(run_dir: Path) -> PasquaziEvolution:
    with np.load(run_dir / _EVOLUTION_FILENAME, allow_pickle=False) as data:
        ring = FieldEvolution(data["theta"], data["times"], data["fields"])
        return PasquaziEvolution(ring, data["loop"])
