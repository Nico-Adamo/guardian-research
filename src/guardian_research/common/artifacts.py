"""RunWriter: the single place that writes a run's structured artifacts.

A run directory is self-describing and ingestable from anywhere:

    runs/<experiment>/<run_id>/
        results.json     # the RunResult (final source of truth)
        metrics.jsonl    # streamed during training (one JSON object per line)
        config.yaml      # the exact resolved Hydra config
        env.json         # git + environment provenance
        artifacts/       # plots, checkpoint metadata, etc.

The same writer is used on a laptop CPU smoke test and on a disposable GPU
worker; ``collect`` simply rsyncs these directories back.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .env_info import env_info, git_info
from .paths import ensure_dir, run_dir
from .schemas import MetricPoint, RunResult


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(prefix: str = "") -> str:
    """Time-ordered, human-readable, collision-resistant run id."""
    import uuid

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix + '-' if prefix else ''}{ts}-{suffix}"


class RunWriter:
    def __init__(self, experiment: str, run_id: str, seed: int):
        self.dir = ensure_dir(run_dir(experiment, run_id))
        ensure_dir(self.dir / "artifacts")
        self.result = RunResult(
            run_id=run_id,
            experiment=experiment,
            seed=seed,
            status="running",
            created_at=utcnow_iso(),
            git=git_info(),
            env=env_info(),
        )
        self._metrics_fp = open(self.dir / "metrics.jsonl", "w")

    # -- configuration / params -------------------------------------------- #
    def set_config(self, config: dict[str, Any]) -> None:
        self.result.config = config
        (self.dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    def set_params(self, **params: Any) -> None:
        self.result.params.update(params)

    # -- metrics ----------------------------------------------------------- #
    def log_metric(self, name: str, value: float, step: int) -> None:
        value = float(value)
        self.result.metrics.setdefault(name, []).append(MetricPoint(step=step, value=value))
        self._metrics_fp.write(json.dumps({"step": step, "name": name, "value": value}) + "\n")
        self._metrics_fp.flush()

    def log_metrics(self, step: int, **metrics: float) -> None:
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def set_final(self, **metrics: float) -> None:
        self.result.final_metrics.update({k: float(v) for k, v in metrics.items()})

    # -- artifacts --------------------------------------------------------- #
    def artifact_path(self, name: str) -> Path:
        return self.dir / "artifacts" / name

    def add_artifact(self, name: str) -> Path:
        path = self.artifact_path(name)
        rel = str(path.relative_to(self.dir))
        if rel not in self.result.artifacts:
            self.result.artifacts.append(rel)
        return path

    # -- lifecycle --------------------------------------------------------- #
    def _write_aux(self) -> None:
        (self.dir / "env.json").write_text(
            json.dumps({"git": self.result.git.model_dump(), "env": self.result.env.model_dump()}, indent=2)
        )

    def finish(self, status: str = "completed", error: str | None = None, notes: str | None = None) -> Path:
        self.result.status = status  # type: ignore[assignment]
        self.result.finished_at = utcnow_iso()
        if error:
            self.result.error = error
        if notes:
            self.result.notes = notes
        try:
            self._metrics_fp.close()
        except Exception:
            pass
        self._write_aux()
        out = self.dir / "results.json"
        out.write_text(self.result.model_dump_json(indent=2))
        return out

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.finish(status="failed", error=f"{exc_type.__name__}: {exc}")
        elif self.result.status == "running":
            self.finish(status="completed")


def load_result(path: str | Path) -> RunResult:
    path = Path(path)
    return RunResult.model_validate_json(path.read_text())
