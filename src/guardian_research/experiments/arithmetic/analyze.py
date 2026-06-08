"""Arithmetic-specific analysis: the 'curves cross' check.

The catapult prediction is NOT 'lower final loss'; it is that a high-LR / cyclic
recipe eventually beats the baseline on the **hard** split. This module turns
that into a falsifiable, mechanical comparison so a report can state plainly
whether the predicted crossover is present in the runs at hand — without
overclaiming.
"""

from __future__ import annotations

from ...common.schemas import RunResult

BASELINE_SCHEDULES = {"baseline_cosine"}


def _final(r: RunResult, key: str) -> float | None:
    if key in r.final_metrics:
        return r.final_metrics[key]
    series = r.metrics.get(key)
    return series[-1].value if series else None


def crossover_summary(results: list[RunResult]) -> str:
    if not results:
        return "_No runs to analyze for crossover._\n"

    baselines = [r for r in results if r.params.get("schedule") in BASELINE_SCHEDULES]
    others = [r for r in results if r.params.get("schedule") not in BASELINE_SCHEDULES]

    lines = ["## Catapult crossover check (hard split)", ""]
    if not baselines:
        lines.append("_No `baseline_cosine` control run present — cannot evaluate a crossover._\n")
        return "\n".join(lines)

    def best_hard(rs: list[RunResult]) -> tuple[float, RunResult | None]:
        best, who = -1.0, None
        for r in rs:
            h = _final(r, "final_hard_acc") or _final(r, "hard_acc")
            if h is not None and h > best:
                best, who = h, r
        return best, who

    base_hard, base_run = best_hard(baselines)
    other_hard, other_run = best_hard(others)

    lines.append(f"- best baseline hard-accuracy: **{base_hard:.3f}**"
                 + (f" ({base_run.params.get('schedule')}/seed{base_run.seed})" if base_run else ""))
    if other_run is not None:
        lines.append(f"- best non-baseline hard-accuracy: **{other_hard:.3f}**"
                     f" ({other_run.params.get('schedule')}/seed{other_run.seed})")
        delta = other_hard - base_hard
        verdict = (
            "a non-baseline recipe is **ahead** on the hard split"
            if delta > 0
            else "the baseline is **not yet** beaten on the hard split"
        )
        lines.append(f"- Δ(best non-baseline − best baseline) = **{delta:+.3f}** → {verdict}.")
    else:
        lines.append("- _No non-baseline schedule runs present to compare._")

    lines.append("")
    lines.append("> Interpretation guard: this is a small-scale, single-report snapshot. "
                 "A real crossover claim (per H001) requires matched compute, multiple seeds, "
                 "and the pre-registered stop conditions in `planning/hypotheses/H001-catapult-arithmetic.md`.")
    lines.append("")
    return "\n".join(lines)
