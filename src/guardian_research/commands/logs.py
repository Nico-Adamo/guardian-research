"""`ga logs <run_id>` — show a run's location and tail its metrics stream."""

from __future__ import annotations

from ..common.logging import console
from ..common.paths import runs_dir

NAME = "logs"
HELP = "Show a run's artifacts and tail its metrics.jsonl: ga logs <run_id>"


def _find_run(run_id: str):
    base = runs_dir()
    if not base.exists():
        return None
    for results in base.glob(f"*/{run_id}/results.json"):
        return results.parent
    # allow suffix match (the short id printed in reports)
    for results in base.glob("*/*/results.json"):
        if results.parent.name.endswith(run_id):
            return results.parent
    return None


def run(argv: list[str]) -> int:
    if not argv:
        console.print("usage: [bold]ga logs <run_id>[/bold]")
        return 2
    run_id = argv[0]
    run_path = _find_run(run_id)
    if run_path is None:
        console.print(f"[red]no run matching[/red] {run_id} under {runs_dir()}")
        return 1
    console.print(f"[bold]run dir:[/bold] {run_path}")
    for f in ("results.json", "config.yaml", "env.json"):
        p = run_path / f
        if p.exists():
            console.print(f"  - {f} ({p.stat().st_size} bytes)")
    metrics = run_path / "metrics.jsonl"
    if metrics.exists():
        lines = metrics.read_text().splitlines()
        console.print(f"\n[bold]metrics.jsonl[/bold] (last 15 of {len(lines)}):")
        for line in lines[-15:]:
            console.print(f"  {line}")
    return 0
