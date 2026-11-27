import subprocess
from pathlib import Path


class Git:
    @staticmethod
    def _run(*args: str, cwd: Path) -> str:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
        ).strip()

    @staticmethod
    def commit(cwd: Path) -> str:
        return Git._run("rev-parse", "HEAD", cwd=cwd)

    @staticmethod
    def is_dirty(cwd: Path) -> bool:
        return bool(
            Git._run(
                "status",
                "--porcelain",
                "--untracked-files=all",
                cwd=cwd,
            )
        )

    @staticmethod
    def require_clean(cwd: Path) -> None:
        if Git.is_dirty(cwd):
            raise RuntimeError("Git working tree must be clean")
