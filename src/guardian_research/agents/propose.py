"""Propose the next sweep from prior run metrics — but never launch it.

This is the core of the agent autonomy model: an agent may *read results and
draft a proposal*; it may not spend money. The proposal is a YAML artifact that
must pass ``ga validate-proposal`` (budget + data class + config validity +
reproducibility) and then explicit human approval before any launch.

The drafting here is deliberately simple and transparent (a couple of
heuristics over the run table), not a black box — you can always read the
proposal and see exactly what it wants to run and why.
"""

from __future__ import annotations

from ..common.budget import BudgetPolicy, estimate_cost
from ..common.env_info import current_sha, git_info
from ..common.schemas import Proposal
from ..tracking.ingest import load_runs

# Default arithmetic sweep axes (Hydra override keys).
# baseline_cosine is INCLUDED in the schedule axis on purpose: the crossover test
# must compare the catapult candidates against a control trained at the *same*
# commit, scale, and seeds (matched compute), not against stale local runs.
DEFAULT_AXES: dict[str, list] = {
    "schedule": ["baseline_cosine", "onecycle_high_lr", "cyclic_lr", "cyclic_weight_decay"],
    "train.lr": [5.0e-4, 1.0e-3, 2.0e-3],
    "train.weight_decay": [0.01, 0.1],
}
DEFAULT_SEEDS = [0, 1]


def _shrink_to_budget(
    axes: dict[str, list], seeds: list[int], budget_usd: float, hours_per_job: float, gpu: str, per_job_cap: float
) -> tuple[dict[str, list], list[int], int, float, float]:
    """Trim the grid until total estimated cost fits the sweep budget."""
    axes = {k: list(v) for k, v in axes.items()}
    seeds = list(seeds)

    def grid() -> int:
        n = 1
        for v in axes.values():
            n *= max(1, len(v))
        return n * max(1, len(seeds))

    per_job = estimate_cost(1, hours_per_job=hours_per_job, gpu=gpu)
    # Trim largest axis / seeds until total fits.
    guard = 0
    while grid() * per_job > budget_usd and guard < 50:
        guard += 1
        if len(seeds) > 1:
            seeds = seeds[:-1]
            continue
        # shrink the longest axis
        longest = max(axes, key=lambda k: len(axes[k]))
        if len(axes[longest]) > 1:
            axes[longest] = axes[longest][:-1]
        else:
            break
    total = round(grid() * per_job, 2)
    return axes, seeds, grid(), per_job, total


def propose_sweep(experiment: str, budget_usd: float, name: str | None = None) -> Proposal:
    policy = BudgetPolicy.load()
    results = load_runs(experiment)

    # Data-driven rationale: report the current best hard-split accuracy if any.
    best_hard = None
    for r in results:
        h = r.final_metrics.get("final_hard_acc")
        if h is not None and (best_hard is None or h > best_hard):
            best_hard = h
    rationale = (
        f"{len(results)} prior run(s); current best hard-split accuracy = "
        f"{best_hard:.3f}." if best_hard is not None
        else "No prior runs found; proposing an initial schedule/lr/wd sweep."
    )

    gpu = "l40s"
    hours_per_job = 0.25
    per_job_cap = policy.max_job_cost_usd
    axes, seeds, grid, per_job, total = _shrink_to_budget(
        DEFAULT_AXES, DEFAULT_SEEDS, budget_usd, hours_per_job, gpu, per_job_cap
    )

    return Proposal(
        name=name or f"{experiment}_sweep_v_next",
        experiment=experiment,
        hypothesis=(
            "H001: high-LR / cyclic schedules (one-cycle, cyclic LR, cyclic WD) improve "
            "HARD arithmetic accuracy relative to a baseline_cosine control at matched compute."
        ),
        expected_signal=(
            "A non-baseline schedule's final hard-split accuracy exceeds the baseline's "
            "by a margin that survives across seeds ('the curves cross' on the hard split)."
        ),
        metric="final_hard_acc",
        ablation=(
            "baseline_cosine is the control; vary lr x weight_decay x schedule x seed to "
            "isolate which factor (peak LR, WD cycling) drives any hard-split gain."
        ),
        stop_conditions=[
            "Stop a shard early if train_loss diverges (NaN) or train_acc stays < 0.05 after 50% of steps.",
            "Stop the sweep if no non-baseline schedule beats baseline hard_acc after all seeds (downgrade H001).",
            f"Hard total-cost ceiling: {total} USD (per-job <= {per_job_cap} USD).",
        ],
        base_config="+exp=arithmetic_catapult model=tiny_transformer",
        sweep=axes,
        seeds=seeds,
        data_class="synthetic",
        provider=policy.allowed_providers[0] if policy.allowed_providers else "runpod",
        gpu=gpu,
        hours_per_job=hours_per_job,
        per_job_cost_usd=per_job,
        estimated_cost_usd=total,
        max_cost_usd=min(budget_usd, policy.max_daily_cost_usd),
        created_from_runs=[r.run_id for r in results],
        reproducibility={
            "git_sha": current_sha(),
            "git_dirty": git_info().dirty,
            "requires_clean_tree": policy.require_clean_git_tree,
            "requires_exact_sha": policy.require_exact_commit_sha,
            "config": "+exp=arithmetic_catapult model=tiny_transformer",
            "grid_size": grid,
        },
        rationale=rationale,
    )
