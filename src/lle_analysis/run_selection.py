import csv
from pathlib import Path

from lle_runs.runs import Run, get_run


def load_runs_from_csv(path: Path) -> list[Run]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return [get_run(row["run_id"]) for row in csv.DictReader(source)]
