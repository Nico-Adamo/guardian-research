"""Local launcher: run an experiment in-process on this machine (CPU/GPU).

The control plane uses this for smoke tests and tiny probes. It dispatches on
``cfg['runner']`` to the right experiment module. Cloud launching is a separate,
gated path (see launchers/skypilot.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

RUNNERS: dict[str, str] = {
    "arithmetic": "guardian_research.experiments.arithmetic.train",
    "modular": "guardian_research.experiments.modular.train",
    "dynamic_grokking": "guardian_research.experiments.dynamic_grokking.run",
    "persona": "guardian_research.experiments.persona.train_persona",
}


def run_local(cfg: dict[str, Any]) -> Path:
    import importlib

    runner = cfg.get("runner")
    if not runner:
        raise ValueError(
            "cfg.runner is not set. Did you pass an experiment, e.g. `+exp=arithmetic_catapult`?"
        )
    module_path = RUNNERS.get(runner)
    if module_path is None:
        raise ValueError(f"unknown runner '{runner}'. Known: {sorted(RUNNERS)}")
    module = importlib.import_module(module_path)
    if not hasattr(module, "run"):
        raise AttributeError(f"runner module {module_path} has no run(cfg) function")
    return module.run(cfg)
