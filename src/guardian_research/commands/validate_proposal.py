"""`ga validate-proposal` — gate a proposal on budget/data/config/reproducibility."""

from __future__ import annotations

import argparse

from ..agents.validate_proposal import validate_proposal_file
from ..common.logging import console

NAME = "validate-proposal"
HELP = "Validate a proposal YAML (budget, data class, config validity, reproducibility)"


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ga validate-proposal", add_help=True)
    ap.add_argument("path")
    args = ap.parse_args(argv)

    report = validate_proposal_file(args.path)

    from rich.table import Table

    t = Table(title=f"validation: {report.target}")
    t.add_column("check")
    t.add_column("ok")
    t.add_column("detail", overflow="fold")
    for c in report.checks:
        style = "green" if c.passed else "red"
        t.add_row(c.name, f"[{style}]{'✓' if c.passed else '✗'}[/{style}]", c.detail)
    console.print(t)

    if report.passed:
        console.print("[green]✓ proposal PASSES — eligible for human approval and a gated launch.[/green]")
        return 0
    console.print("[red]✗ proposal FAILS validation — must not be launched.[/red]")
    return 1
