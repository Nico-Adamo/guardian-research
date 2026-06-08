"""`ga analyze` — render a local markdown report (with plots/tables) from runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..common.logging import console
from ..common.paths import reports_dir
from ..tracking.ingest import load_runs
from ..tracking.reports import generate_experiment_report

NAME = "analyze"
HELP = "Render a markdown report from collected runs: --experiment NAME [--since 7d] [--write PATH]"


def _parse_since(s: str | None) -> float | None:
    if not s:
        return None
    s = s.strip().lower()
    try:
        if s.endswith("d"):
            return float(s[:-1])
        if s.endswith("h"):
            return float(s[:-1]) / 24.0
        return float(s)
    except ValueError:
        return None


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="ga analyze", add_help=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--since", default=None, help="e.g. 7d or 12h")
    ap.add_argument("--write", default=None, help="output markdown path")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    results = load_runs(args.experiment, since_days=_parse_since(args.since))
    out_path = Path(args.write) if args.write else (reports_dir() / "runs" / f"{args.experiment}.md")

    md = generate_experiment_report(args.experiment, results, out_path, title=args.title)

    # Experiment-specific addenda (e.g., arithmetic 'curves cross' check).
    if args.experiment == "arithmetic_catapult":
        from ..experiments.arithmetic.analyze import crossover_summary

        addendum = "\n" + crossover_summary(results)
        out_path.write_text(md + addendum)
        md = md + addendum

    console.print(f"[green]✓ wrote report[/green] ({len(results)} runs) → {out_path}")
    return 0
