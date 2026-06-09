"""`ga approve` — record human sign-off on a validated proposal.

This is step 3 of the loop (propose -> validate -> APPROVE -> launch). It refuses
to approve a proposal that does not pass validation, and binds the approval to the
proposal's content hash + the current commit (see agents/approval.py).
"""

from __future__ import annotations

import argparse

from ..agents.approval import write_approval
from ..common.logging import console

NAME = "approve"
HELP = "Record human approval of a validated proposal: ga approve PATH [--by NAME] [--note ...]"


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ga approve", add_help=True)
    ap.add_argument("path")
    ap.add_argument("--by", default="", help="who is approving (required for a real sign-off)")
    ap.add_argument("--note", default="")
    args = ap.parse_args(argv)

    if not args.by:
        console.print("[red]--by is required[/red] (approval must name a human, e.g. --by alice).")
        return 2
    try:
        rec = write_approval(args.path, by=args.by, note=args.note)
    except ValueError as exc:
        console.print(f"[red]✗ refused:[/red] {exc}")
        console.print("Fix the proposal and re-validate before approving.")
        return 1
    console.print(f"[green]✓ approved[/green] {args.path}")
    console.print(f"  by {rec['approved_by']} @ commit {rec['git_sha'][:10]} ({rec['approved_at']})")
    console.print(f"  bound to proposal sha256 {rec['proposal_sha256'][:16]}…")
    console.print("[dim]Now launch (gated):[/dim] "
                  f"[bold]ga launch --dry-run --proposal {args.path}[/bold] then add --yes")
    return 0
