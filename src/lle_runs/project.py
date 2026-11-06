from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"pyproject.toml не найден выше {path}")


def get_experiments_dir() -> Path:
    return find_project_root() / "experiments"


def get_experiment_dir(number: int) -> Path:
    if not isinstance(number, int) or isinstance(number, bool) or number < 1:
        raise ValueError("Ожидается положительный номер эксперимента")
    matches = sorted(p for p in get_experiments_dir().glob(f"{number:02d}_*") if p.is_dir())
    if not matches:
        raise FileNotFoundError(f"Эксперимент {number:02d} не найден")
    if len(matches) != 1:
        raise ValueError(f"Номер эксперимента {number:02d} неоднозначен: {matches}")
    return matches[0]


def get_runs_dir(start: Path | None = None) -> Path:
    return find_project_root(start) / "runs"
