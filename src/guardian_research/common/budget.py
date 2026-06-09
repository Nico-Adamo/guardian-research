"""Budget + autonomy guard — the spending safety gate.

Policy lives in ``pyproject.toml`` under ``[tool.guardian]`` (single source of
truth) and can be tightened (never loosened silently) by environment variables.
This module never *spends* anything; it answers "is this allowed?" and records
intent in a local ledger. Actually launching is the launcher's job, and only
after these checks pass *and* a human passes ``--yes``.

Autonomy tiers (from planning/guardian/planning.md):

* Tier 0 — unrestricted: docs, tests, CPU runs, summarizing, *proposing*. No money.
* Tier 1 — bounded automation: small GPU jobs, allowed only if ALL gates pass.
* Tier 2 — approval required: anything bigger / private data / uploads / deletes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import tomllib

from .paths import repo_root, runs_dir
from .schemas import Proposal, ValidationReport

# Approximate on-demand GPU pricing (USD/hour), from the research-program notes.
# These are used for *estimates only*; the real bill comes from the provider.
GPU_HOURLY_USD: dict[str, float] = {
    "a10": 0.60,
    "l4": 0.50,
    "rtx4090": 0.44,
    "a40": 0.44,
    "l40s": 0.86,
    "a10g": 0.75,
    "a100": 1.39,
    "a100-80gb": 1.79,
    "h100": 3.29,
    "cpu": 0.10,
}

DEFAULT_GPU = "l40s"


@dataclass
class BudgetPolicy:
    max_job_cost_usd: float = 5.0
    max_daily_cost_usd: float = 25.0
    total_budget_usd: float = 2000.0
    allowed_providers: list[str] = field(default_factory=lambda: ["runpod", "lambda", "modal"])
    allowed_data_classes: list[str] = field(default_factory=lambda: ["public", "synthetic"])
    allow_private_persona_data: bool = False
    require_clean_git_tree: bool = True
    require_exact_commit_sha: bool = True
    require_dry_run_first: bool = True

    @classmethod
    def load(cls) -> BudgetPolicy:
        """Load from pyproject.toml, then apply (tightening) env overrides."""
        data: dict[str, Any] = {}
        pyproject = repo_root() / "pyproject.toml"
        if pyproject.exists():
            with open(pyproject, "rb") as f:
                data = tomllib.load(f).get("tool", {}).get("guardian", {})
        policy = cls(
            max_job_cost_usd=float(data.get("max_job_cost_usd", cls.max_job_cost_usd)),
            max_daily_cost_usd=float(data.get("max_daily_cost_usd", cls.max_daily_cost_usd)),
            total_budget_usd=float(data.get("total_budget_usd", cls.total_budget_usd)),
            allowed_providers=list(data.get("allowed_providers", ["runpod", "lambda", "modal"])),
            allowed_data_classes=list(data.get("allowed_data_classes", ["public", "synthetic"])),
            allow_private_persona_data=bool(data.get("allow_private_persona_data", False)),
            require_clean_git_tree=bool(data.get("require_clean_git_tree", True)),
            require_exact_commit_sha=bool(data.get("require_exact_commit_sha", True)),
            require_dry_run_first=bool(data.get("require_dry_run_first", True)),
        )
        # Env overrides can only *lower* the spending ceilings (defense in depth).
        if (env := os.environ.get("GUARDIAN_MAX_JOB_COST_USD")):
            policy.max_job_cost_usd = min(policy.max_job_cost_usd, float(env))
        if (env := os.environ.get("GUARDIAN_MAX_DAILY_COST_USD")):
            policy.max_daily_cost_usd = min(policy.max_daily_cost_usd, float(env))
        return policy


def estimate_cost(
    grid_size: int,
    hours_per_job: float = 0.25,
    gpu: str = DEFAULT_GPU,
    num_gpus: int = 1,
) -> float:
    rate = GPU_HOURLY_USD.get(gpu.lower(), GPU_HOURLY_USD[DEFAULT_GPU])
    return round(grid_size * hours_per_job * rate * max(1, num_gpus), 2)


# --------------------------------------------------------------------------- #
# Local daily-spend ledger (control-plane state, gitignored under runs/)        #
# --------------------------------------------------------------------------- #
def _ledger_path() -> Path:
    return runs_dir() / "_budget_ledger.json"


def _load_ledger() -> dict[str, float]:
    path = _ledger_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def spent_today() -> float:
    return float(_load_ledger().get(date.today().isoformat(), 0.0))


def record_spend(amount_usd: float) -> None:
    ledger = _load_ledger()
    today = date.today().isoformat()
    ledger[today] = round(ledger.get(today, 0.0) + float(amount_usd), 2)
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2))


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
class BudgetGuard:
    def __init__(self, policy: BudgetPolicy | None = None):
        self.policy = policy or BudgetPolicy.load()

    def validate_proposal(self, proposal: Proposal) -> ValidationReport:
        """Validate the *budget + data-class + provider* facets of a proposal.

        Config-validity and reproducibility checks live in
        ``agents.validate_proposal`` which composes this with a Hydra compose
        check; this method is the money/data half.
        """
        p = self.policy
        r = ValidationReport.start(proposal.name)

        r.add(
            "cost_estimate_positive",
            proposal.estimated_cost_usd > 0 and proposal.per_job_cost_usd > 0,
            f"per_job={proposal.per_job_cost_usd} total={proposal.estimated_cost_usd}",
        )
        r.add(
            "per_job_within_policy",
            proposal.per_job_cost_usd <= p.max_job_cost_usd,
            f"per_job_cost_usd={proposal.per_job_cost_usd} <= policy max_job_cost_usd={p.max_job_cost_usd}",
        )
        r.add(
            "total_within_max",
            proposal.estimated_cost_usd <= proposal.max_cost_usd,
            f"total {proposal.estimated_cost_usd} <= sweep budget {proposal.max_cost_usd}",
        )
        projected_daily = spent_today() + proposal.estimated_cost_usd
        r.add(
            "within_daily_budget",
            projected_daily <= p.max_daily_cost_usd,
            f"spent_today + total = {round(projected_daily, 2)} <= daily cap {p.max_daily_cost_usd}",
        )
        r.add(
            "data_class_allowed",
            proposal.data_class in p.allowed_data_classes,
            f"data_class={proposal.data_class}, allowed={p.allowed_data_classes}",
        )
        if proposal.data_class == "private":
            r.add(
                "private_data_permitted",
                p.allow_private_persona_data,
                "private persona data requires allow_private_persona_data=true + approval gate",
            )
        r.add(
            "provider_allowed",
            proposal.provider in p.allowed_providers,
            f"provider={proposal.provider}, allowed={p.allowed_providers}",
        )
        return r

    def preflight_launch(
        self,
        *,
        provider: str,
        data_class: str,
        per_job_cost_usd: float,
        total_cost_usd: float,
        max_total_cost_usd: float,
        git_dirty: bool,
        git_sha: str | None,
        dry_run_done: bool,
    ) -> ValidationReport:
        """Hard gate run immediately before a real (money-spending) launch."""
        p = self.policy
        r = ValidationReport.start(f"launch:{provider}")
        r.add("provider_allowed", provider in p.allowed_providers, f"{provider} in {p.allowed_providers}")
        r.add("data_class_allowed", data_class in p.allowed_data_classes, f"{data_class} in {p.allowed_data_classes}")
        r.add(
            "per_job_within_cap",
            per_job_cost_usd <= p.max_job_cost_usd,
            f"per-job ${per_job_cost_usd} <= job cap ${p.max_job_cost_usd}",
        )
        r.add(
            "total_within_max",
            total_cost_usd <= max_total_cost_usd,
            f"total ${total_cost_usd} <= declared max ${max_total_cost_usd}",
        )
        r.add(
            "within_daily_budget",
            spent_today() + total_cost_usd <= p.max_daily_cost_usd,
            f"{round(spent_today() + total_cost_usd, 2)} <= daily cap ${p.max_daily_cost_usd}",
        )
        if p.require_clean_git_tree:
            r.add("clean_git_tree", not git_dirty, "working tree must be clean before a cloud launch")
        if p.require_exact_commit_sha:
            r.add("exact_commit_sha", bool(git_sha) and git_sha != "unknown", f"git_sha={git_sha}")
        if p.require_dry_run_first:
            r.add("dry_run_first", dry_run_done, "a --dry-run must precede a real launch")
        return r
