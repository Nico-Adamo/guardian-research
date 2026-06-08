"""Validate a proposal before it can be approved/launched.

Four facets, all required to pass:
  1. budget    — per-job + total cost within policy caps (BudgetGuard);
  2. data class — only allowed classes; private requires explicit permission;
  3. config     — the base_config + first sweep point actually composes in Hydra;
  4. reproducibility + scientific framing — git SHA recorded, and the proposal
     carries a hypothesis, metric, expected signal, ablation, and stop conditions.

This is what stands between "an agent had an idea" and "money was spent".
"""

from __future__ import annotations

from pathlib import Path

import yaml

from ..common.budget import BudgetGuard, estimate_cost
from ..common.hydra_utils import try_compose
from ..common.schemas import Proposal, ValidationReport


def load_proposal(path: str | Path) -> Proposal:
    data = yaml.safe_load(Path(path).read_text())
    return Proposal.model_validate(data)


def validate_proposal(proposal: Proposal) -> ValidationReport:
    # 1. Budget / data class / provider.
    report = BudgetGuard().validate_proposal(proposal)

    # 2. Required scientific framing (no "just try stuff" proposals).
    report.add("has_hypothesis", bool(proposal.hypothesis.strip()), "hypothesis present")
    report.add("has_metric", bool(proposal.metric.strip()), f"metric={proposal.metric!r}")
    report.add("has_expected_signal", bool(proposal.expected_signal.strip()), "expected signal present")
    report.add("has_ablation", bool(proposal.ablation.strip()), "ablation present")
    report.add("has_stop_conditions", len(proposal.stop_conditions) > 0,
               f"{len(proposal.stop_conditions)} stop condition(s)")

    # 3. Config validity — compose base_config + the first point of each axis.
    overrides = proposal.base_config.split()
    for key, values in proposal.sweep.items():
        if values:
            v = values[0]
            overrides.append(f"{key}={v}")
    if proposal.seeds:
        overrides.append(f"seed={proposal.seeds[0]}")
    ok, msg = try_compose(overrides)
    report.add("config_composes", ok, msg)

    # 4. Reproducibility.
    sha = proposal.reproducibility.get("git_sha")
    report.add("has_git_sha", bool(sha) and sha != "unknown", f"git_sha={sha}")

    # 5. Cost internal consistency: declared total ~= per-job * grid.
    recomputed = estimate_cost(proposal.grid_size(), hours_per_job=proposal.hours_per_job, gpu=proposal.gpu)
    consistent = abs(recomputed - proposal.estimated_cost_usd) <= max(0.5, 0.2 * recomputed)
    report.add(
        "cost_matches_grid",
        consistent,
        f"recomputed total={recomputed} vs declared={proposal.estimated_cost_usd} (grid={proposal.grid_size()})",
    )
    return report


def validate_proposal_file(path: str | Path) -> ValidationReport:
    return validate_proposal(load_proposal(path))
