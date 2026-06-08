"""Ingest run artifacts written anywhere (laptop or cloud worker) into memory.

The control plane reads ``runs/<experiment>/*/results.json`` — that's the whole
contract. ``ga collect`` rsyncs a worker's run directory here; ingestion does
not care where it came from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..common.artifacts import load_result
from ..common.paths import runs_dir
from ..common.schemas import RunResult


def load_runs(experiment: Optional[str] = None, since_days: Optional[float] = None) -> list[RunResult]:
    base = runs_dir()
    if not base.exists():
        return []
    exp_dirs = [base / experiment] if experiment else [d for d in base.iterdir() if d.is_dir()]
    results: list[RunResult] = []
    for exp_dir in exp_dirs:
        if not exp_dir.is_dir():
            continue
        for results_json in sorted(exp_dir.glob("*/results.json")):
            try:
                results.append(load_result(results_json))
            except Exception:
                continue
    if since_days is not None:
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        results = [r for r in results if _created(r) >= cutoff]
    return results


def _created(r: RunResult):
    from datetime import datetime

    try:
        return datetime.fromisoformat(r.created_at)
    except Exception:
        from datetime import timezone

        return datetime.min.replace(tzinfo=timezone.utc)


def runs_dataframe(results: list[RunResult]) -> pd.DataFrame:
    """Flatten runs into a comparison table: one row per run."""
    rows = []
    for r in results:
        row = {
            "run_id": r.run_id,
            "experiment": r.experiment,
            "status": r.status,
            "seed": r.seed,
            "schedule": r.params.get("schedule"),
            "model": r.params.get("model"),
            "git_sha": r.git.sha[:10],
            "git_dirty": r.git.dirty,
            "created_at": r.created_at,
        }
        row.update({f"p.{k}": v for k, v in r.params.items()})
        row.update({f"m.{k}": v for k, v in r.final_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)
