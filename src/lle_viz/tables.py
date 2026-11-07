import csv
from collections.abc import Iterable, Mapping
from pathlib import Path


def format_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return "; ".join(map(format_value, value))
    return str(value)


def table_markdown(rows: Iterable[Mapping[str, object]]) -> str:
    rows = list(rows)
    if not rows:
        return ""
    headers = list(rows[0])
    values = [headers, ["---"] * len(headers)]
    values.extend([[row[key] for key in headers] for row in rows])
    return "\n".join(
        "| " + " | ".join(format_value(v).replace("|", r"\|") for v in row) + " |"
        for row in values
    ) + "\n"


def save_table(results_dir: Path, name: str, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with (results_dir / f"{name}.csv").open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (results_dir / f"{name}.md").write_text(table_markdown(rows), encoding="utf-8")
