import csv
from pathlib import Path

from lle.runs.runs import Run, get_run


def load_runs_from_csv(path: Path) -> list[Run]:
    """Загружает запуски в порядке идентификаторов в CSV."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        return [get_run(row["run_id"]) for row in csv.DictReader(source)]
