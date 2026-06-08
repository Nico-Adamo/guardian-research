"""`ga status` — local control-plane status: budget ledger + recent runs."""

from __future__ import annotations

from collections import Counter

from ..common.budget import BudgetPolicy, spent_today
from ..common.logging import console
from ..tracking.ingest import load_runs

NAME = "status"
HELP = "Show budget ledger and a summary of local runs"


def run(argv: list[str]) -> int:
    policy = BudgetPolicy.load()
    console.print("[bold]Budget[/bold]")
    console.print(f"  spent today      : ${spent_today():.2f}")
    console.print(f"  daily cap        : ${policy.max_daily_cost_usd:.2f}")
    console.print(f"  per-job cap      : ${policy.max_job_cost_usd:.2f}")
    console.print(f"  total PoC budget : ${policy.total_budget_usd:.2f}")
    console.print(f"  allowed providers: {policy.allowed_providers}")
    console.print(f"  allowed data     : {policy.allowed_data_classes} (private allowed: {policy.allow_private_persona_data})")

    runs = load_runs()
    console.print("\n[bold]Runs[/bold]")
    if not runs:
        console.print("  (none — run `ga train +exp=arithmetic_catapult ...` or `just smoke`)")
        return 0
    by_exp = Counter(r.experiment for r in runs)
    by_status = Counter(r.status for r in runs)
    console.print(f"  total: {len(runs)} | by status: {dict(by_status)}")
    for exp, n in by_exp.most_common():
        console.print(f"  - {exp}: {n}")
    return 0
