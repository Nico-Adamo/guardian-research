"""Canonical filesystem layout for the experiment factory.

Everything that matters lives in the repo (the source of truth). Local, cheaply
re-creatable caches (runs/, artifacts/, mlruns/, data/) are gitignored but have a
fixed, documented structure so that results written by a disposable GPU worker
can be ingested locally without any hand-edited state.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root.

    Resolution order:
    1. ``GUARDIAN_REPO_ROOT`` env var (set by cloud workers / tests).
    2. Walk up from this file looking for a ``pyproject.toml``.
    3. Fall back to the current working directory.
    """
    env = os.environ.get("GUARDIAN_REPO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def runs_dir() -> Path:
    return repo_root() / "runs"


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def mlruns_dir() -> Path:
    return repo_root() / "mlruns"


def conf_dir() -> Path:
    return repo_root() / "conf"


def reports_dir() -> Path:
    return repo_root() / "reports"


def data_dir() -> Path:
    return repo_root() / "data"


def run_dir(experiment: str, run_id: str) -> Path:
    """Directory holding all artifacts for a single run.

    Layout: ``runs/<experiment>/<run_id>/{results.json,metrics.jsonl,config.yaml,env.json,artifacts/}``
    """
    return runs_dir() / experiment / run_id


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
