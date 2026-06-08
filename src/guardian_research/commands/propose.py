"""`ga propose` — draft a next-sweep proposal YAML from prior runs (no launch)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ..agents.propose import propose_sweep
from ..common.logging import console
from ..common.paths import reports_dir

NAME = "propose"
HELP = "Draft a next-sweep proposal: --experiment NAME --budget-usd N [--write PATH] [--name NAME]"


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ga propose", add_help=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--write", default=None)
    ap.add_argument("--name", default=None)
    args = ap.parse_args(argv)

    proposal = propose_sweep(args.experiment, args.budget_usd, name=args.name)
    out = Path(args.write) if args.write else (reports_dir() / "proposals" / f"{proposal.name}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(proposal.model_dump(), sort_keys=False))

    console.print(f"[green]✓ proposal drafted[/green] → {out}")
    console.print(f"  grid={proposal.grid_size()} jobs | per-job ≈ ${proposal.per_job_cost_usd:.2f} | "
                  f"total ≈ ${proposal.estimated_cost_usd:.2f} (budget ${proposal.max_cost_usd:.2f})")
    console.print(f"  hypothesis: {proposal.hypothesis}")
    console.print("[dim]Validate before any launch:[/dim] "
                  f"[bold]ga validate-proposal {out}[/bold]")
    return 0
