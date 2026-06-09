"""Modular-grokking analysis: did the schedule change the memorize->grok transition?

The H001 question on this testbed is whether a high-LR / cyclic-WD schedule groks
*sooner* or *more reliably* than the baseline at matched compute. This turns that
into a mechanical, honest per-schedule comparison.
"""

from __future__ import annotations

from ...common.schemas import RunResult

BASELINE = "baseline_cosine"


def _f(r: RunResult, key: str) -> float | None:
    if key in r.final_metrics:
        return r.final_metrics[key]
    s = r.metrics.get(key)
    return s[-1].value if s else None


def grokking_summary(results: list[RunResult]) -> str:
    if not results:
        return "_No runs to analyze for grokking._\n"
    lines = ["## Modular-grokking summary (H001 cheap testbed)", ""]
    lines.append("| schedule | seed | final_train | final_val | grok_step | grokked |")
    lines.append("|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda r: (str(r.params.get("schedule")), r.seed)):
        gs = _f(r, "grok_step")
        gs_str = "—" if gs is None or gs < 0 else str(int(gs))
        lines.append(
            f"| {r.params.get('schedule')} | {r.seed} | "
            f"{_f(r, 'final_train_acc'):.3f} | {_f(r, 'final_val_acc'):.3f} | "
            f"{gs_str} | {'yes' if (_f(r, 'grokked') or 0) > 0 else 'no'} |"
        )
    lines.append("")

    base = [r for r in results if r.params.get("schedule") == BASELINE]
    others = [r for r in results if r.params.get("schedule") != BASELINE]
    if base and others:
        def best_val(rs):
            return max((_f(r, "final_val_acc") or 0.0) for r in rs)

        def best_grok(rs):
            steps = [_f(r, "grok_step") for r in rs]
            steps = [s for s in steps if s is not None and s >= 0]
            return min(steps) if steps else None

        bv, ov = best_val(base), best_val(others)
        bg, og = best_grok(base), best_grok(others)
        lines.append(f"- best baseline val_acc **{bv:.3f}** vs best non-baseline **{ov:.3f}** "
                     f"(Δ {ov - bv:+.3f}).")
        if bg is not None or og is not None:
            lines.append(f"- earliest grok step — baseline: {bg if bg is not None else '—'}, "
                         f"non-baseline: {og if og is not None else '—'} "
                         f"(lower = grokked sooner).")
        lines.append("")
    lines.append("> Note: weight-decay schedule multipliers (e.g. `cyclic_weight_decay`'s "
                 "`wd_max_mult`) are **relative to `train.weight_decay`**. The modular default "
                 "uses heavy `base_wd`, so a 20× peak over-regularizes — co-tune them. This is a "
                 "knob to sweep, not a fixed result.")
    lines.append("")
    return "\n".join(lines)
