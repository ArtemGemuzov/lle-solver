from contextlib import contextmanager
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from pydantic import BaseModel
import yaml

from lle_runs.config import C, load_config
from lle_runs.project import find_project_root, get_runs_dir
from lle_runs.git import Git


PROJECT_ROOT = find_project_root(Path(__file__))
RUNS_ROOT = get_runs_dir(Path(__file__))


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Run:
    path: Path

    @property
    def id(self) -> str:
        return self.path.name

    def load_config(self, model: type[C]) -> C:
        return load_config(model, self.path / "parameters.yaml")

    def save_config(self, config: BaseModel) -> None:
        (self.path / "parameters.yaml").write_text(
            yaml.safe_dump(
                config.model_dump(mode="json"),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )

    def load_metadata(self) -> dict:
        return json.loads((self.path / "metadata.json").read_text(encoding="utf-8"))

    def save_metadata(self, metadata: dict) -> None:
        (self.path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def exists(self) -> bool:
        return (self.path / "metadata.json").is_file()


def list_runs(*, script: str | None = None) -> list[Run]:
    if not RUNS_ROOT.exists():
        return []
    runs = []
    for path in sorted(RUNS_ROOT.iterdir()):
        if not (path / "metadata.json").is_file():
            continue
        run = Run(path)
        metadata = run.load_metadata()
        if script is not None and metadata["script"] != script:
            continue
        runs.append(run)
    return runs


def get_run(run_id: str) -> Run:
    if not run_id or Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError("Ожидается ID запуска")
    run = Run(RUNS_ROOT / run_id)
    if not run.exists():
        raise FileNotFoundError(run_id)
    return run


@contextmanager
def create_run(config: BaseModel, *, script: str | Path) -> Generator[Run]:
    started = _now()
    metadata = {
        "script": Path(script).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "git_commit": Git.commit(PROJECT_ROOT),
        "git_dirty": Git.is_dirty(PROJECT_ROOT),
        "started_utc": started.isoformat(),
        "status": "running",
    }

    path = RUNS_ROOT / started.strftime("%Y-%m-%d_%H-%M-%S_%f")
    path.mkdir(parents=True, exist_ok=False)

    run = Run(path)
    run.save_config(config)
    run.save_metadata(metadata)

    try:
        yield run
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    else:
        metadata["status"] = "complete"
    finally:
        metadata["finished_utc"] = _now().isoformat()
        run.save_metadata(metadata)
