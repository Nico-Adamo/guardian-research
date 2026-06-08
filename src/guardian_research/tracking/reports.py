"""Local markdown report generation from collected runs.

Deliberately conservative in tone: a report describes *what was measured*, never
claims a scientific result. Plots are written under ``reports/figures/`` and
referenced with relative paths so the markdown renders on GitHub or locally.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt  # noqa: E402

from ..common.paths import reports_dir  # noqa: E402
from ..common.schemas import RunResult  # noqa: E402
from .ingest import runs_dataframe  # noqa: E402

# Metric groups to plot if present in the runs.
PLOT_GROUPS: dict[str, list[str]] = {
    "loss": ["train_loss", "easy_loss", "hard_loss", "val_loss"],
    "accuracy": ["train_acc", "easy_acc", "hard_acc", "hard_ood_acc", "hard_carry_acc"],
    "memorization_gap": ["memorization_gap"],
    "optim": ["lr", "wd"],
    "grad_norm": ["grad_norm"],
}


def _label(r: RunResult) -> str:
    sched = r.params.get("schedule", "?")
    return f"{sched}/seed{r.seed}/{r.run_id[-6:]}"


def _plot_group(results: list[RunResult], metrics: list[str], title: str, out_png: Path) -> bool:
    present = [m for m in metrics if any(m in r.metrics and r.metrics[m] for r in results)]
    if not present:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for r in results:
        for m in present:
            series = r.metrics.get(m, [])
            if not series:
                continue
            xs = [pt.step for pt in series]
            ys = [pt.value for pt in series]
            ax.plot(xs, ys, label=f"{_label(r)}:{m}", alpha=0.85, linewidth=1.4)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_title(title)
    ax.set_xlabel("step")
    ax.set_ylabel(title)
    ax.legend(fontsize=6, loc="best")
    ax.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return True


def _summary_table(results: list[RunResult]) -> str:
    df = runs_dataframe(results)
    if df.empty:
        return "_No runs found._\n"
    # Keep the table readable: id, schedule, seed, and metric columns.
    metric_cols = [c for c in df.columns if c.startswith("m.")]
    cols = ["run_id", "schedule", "seed", "git_sha", "status", *metric_cols]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    df.columns = [c.replace("m.", "") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == float:
            df[c] = df[c].map(lambda x: f"{x:.4g}" if x == x else "")  # noqa: PLR0124
    return df.to_markdown(index=False) + "\n"


def generate_experiment_report(
    experiment: str,
    results: list[RunResult],
    out_path: Path,
    title: str | None = None,
) -> str:
    """Render a markdown report for ``experiment`` and write figures next to it."""
    out_path = Path(out_path)
    fig_dir = reports_dir() / "figures" / experiment
    title = title or f"Experiment report: {experiment}"

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"- runs ingested: **{len(results)}**")
    if results:
        shas = sorted({r.git.sha[:10] for r in results})
        lines.append(f"- git SHAs present: {', '.join(shas)}")
        if any(r.git.dirty for r in results):
            lines.append("- ⚠️ some runs were produced from a **dirty** git tree (not reproducible)")
    lines.append("")
    lines.append("> This report summarizes *measured tooling output*. It makes no claim "
                 "about whether any hypothesis was confirmed — see `planning/hypotheses/` "
                 "for the claims and their pre-registered stop conditions.")
    lines.append("")

    lines.append("## Run summary")
    lines.append("")
    lines.append(_summary_table(results))

    if results:
        lines.append("## Plots")
        lines.append("")
        for group, metrics in PLOT_GROUPS.items():
            png = fig_dir / f"{group}.png"
            if _plot_group(results, metrics, group, png):
                try:
                    rel = png.relative_to(out_path.parent)
                except ValueError:
                    import os

                    rel = Path(os.path.relpath(png, out_path.parent))
                lines.append(f"### {group}")
                lines.append("")
                lines.append(f"![{group}]({rel.as_posix()})")
                lines.append("")

    md = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    return md
