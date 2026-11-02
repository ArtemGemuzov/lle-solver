from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"pyproject.toml не найден выше {path}")
