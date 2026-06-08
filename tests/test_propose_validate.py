"""The propose -> validate loop: an agent's draft must pass the safety gate."""

from guardian_research.agents.propose import propose_sweep
from guardian_research.agents.validate_proposal import validate_proposal


def test_propose_then_validate_passes():
    proposal = propose_sweep("arithmetic_catapult", budget_usd=25.0)
    assert proposal.estimated_cost_usd > 0
    assert proposal.per_job_cost_usd <= 5.0  # per-job policy cap
    assert proposal.data_class == "synthetic"
    report = validate_proposal(proposal)
    assert report.passed, [(c.name, c.detail) for c in report.checks if not c.passed]


def test_propose_respects_tiny_budget():
    # A very small budget must shrink the grid so the total fits.
    proposal = propose_sweep("arithmetic_catapult", budget_usd=1.0)
    assert proposal.estimated_cost_usd <= 1.0 + 1e-6
    report = validate_proposal(proposal)
    assert report.passed, [(c.name, c.detail) for c in report.checks if not c.passed]
