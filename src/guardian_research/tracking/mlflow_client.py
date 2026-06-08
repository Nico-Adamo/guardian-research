"""Thin MLflow wrapper that logs to a local file store and never crashes a run.

Tracking is local/self-hostable by default (``file:./mlruns``) — no managed SaaS,
in keeping with the security posture for persona data. If MLflow is unavailable
or errors, every call is a no-op so a training run still completes and still
writes its authoritative ``results.json`` via ``RunWriter``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from ..common.logging import get_logger
from ..common.paths import mlruns_dir

log = get_logger(__name__)


def tracking_uri() -> str:
    return f"file:{mlruns_dir()}"


class _NullRun:
    def log_params(self, *a: Any, **k: Any) -> None: ...
    def log_metric(self, *a: Any, **k: Any) -> None: ...
    def log_metrics(self, *a: Any, **k: Any) -> None: ...
    def log_artifact(self, *a: Any, **k: Any) -> None: ...


class MlflowRun:
    def __init__(self, mlflow: Any):
        self._mlflow = mlflow

    def log_params(self, params: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self._mlflow.log_params({k: _scalar(v) for k, v in params.items()})

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        with contextlib.suppress(Exception):
            self._mlflow.log_metric(name, float(value), step=step)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        with contextlib.suppress(Exception):
            self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    def log_artifact(self, path: str) -> None:
        with contextlib.suppress(Exception):
            self._mlflow.log_artifact(path)


def _scalar(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


@contextlib.contextmanager
def start_run(experiment: str, run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Context manager yielding an object with log_params/log_metric(s)/log_artifact."""
    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=run_name, tags=tags or {}):
            yield MlflowRun(mlflow)
    except Exception as exc:  # noqa: BLE001
        log.warning("MLflow disabled (%s); continuing with local results.json only", exc)
        yield _NullRun()
