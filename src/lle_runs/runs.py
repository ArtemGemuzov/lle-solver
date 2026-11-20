import json
import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from pydantic import BaseModel

from lle_runs.config import C, load_config
from lle_runs.git import Git
from lle_runs.project import find_project_root, get_runs_dir

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
        temporary = self.path / "metadata.json.tmp"
        temporary.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(self.path / "metadata.json")

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
        raise ValueError("Expected a run ID")
    run = Run(RUNS_ROOT / run_id)
    if not run.exists():
        raise FileNotFoundError(run_id)
    return run


@contextmanager
def create_run(
    config: BaseModel,
    *,
    script: str | Path,
    extra_metadata: Mapping[str, object] | None = None,
) -> Generator[Run]:
    started = _now()
    metadata = {
        "script": Path(script).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "git_commit": Git.commit(PROJECT_ROOT),
        "git_dirty": Git.is_dirty(PROJECT_ROOT),
        "started_utc": started.isoformat(),
        "status": "running",
    }
    if extra_metadata is not None:
        reserved = metadata.keys() | {"finished_utc", "error"}
        if overlap := reserved.intersection(extra_metadata):
            raise ValueError(f"Reserved metadata fields: {', '.join(sorted(overlap))}")
        metadata.update(extra_metadata)

    path = RUNS_ROOT / started.strftime("%Y-%m-%d_%H-%M-%S_%f")
    path.mkdir(parents=True, exist_ok=False)

    run = Run(path)
    with lock_run(run):
        run.save_config(config)
        with _execute_run(run, metadata):
            yield run


@contextmanager
def lock_run(run: Run) -> Generator[None]:
    """Удерживает системную блокировку файла на время выполнения запуска."""
    with (run.path / "run.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(
                    f"Could not acquire the lock for run {run.id}"
                ) from error
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Run {run.id} is already in use") from error
        # Закрытие файла освобождает блокировку, в том числе при исключении или Ctrl+C.
        yield


@contextmanager
def _execute_run(run: Run, metadata: dict) -> Generator[None]:
    metadata.update(status="running")
    metadata.pop("error", None)
    metadata.pop("finished_utc", None)
    try:
        run.save_metadata(metadata)
        yield
    except BaseException as error:
        metadata.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    else:
        metadata["status"] = "complete"
    finally:
        metadata["finished_utc"] = _now().isoformat()
        current = run.load_metadata() if run.exists() else {}
        current.update(
            {
                key: metadata[key]
                for key in ("status", "finished_utc", "error")
                if key in metadata
            }
        )
        run.save_metadata(current)


@contextmanager
def resume_run(path: Path, *, script: str | Path) -> Generator[Run]:
    """Возобновляет незавершённый запуск под той же блокировкой, что и create_run."""
    run = Run(path.resolve())
    if not run.exists():
        raise FileNotFoundError(f"Run metadata not found: {run.path}")
    with lock_run(run):
        metadata = run.load_metadata()
        expected_script = Path(script).resolve().relative_to(PROJECT_ROOT).as_posix()
        if metadata.get("script") != expected_script:
            raise ValueError("Run belongs to a different script")
        if metadata.get("status") == "running":
            metadata["status"] = "incomplete"
            run.save_metadata(metadata)
        if metadata.get("status") != "incomplete":
            raise ValueError("Only incomplete runs can be resumed")
        with _execute_run(run, metadata):
            yield run
