"""Safety gate: UNSAFE proposals MUST fail validation; a good one MUST pass.

This is the workstream-H counterpart to ``tests/test_budget_guard.py``. Where
that file exercises the budget guard in isolation, here we drive the *full*
``agents.validate_proposal.validate_proposal`` path (budget + data class +
provider + scientific framing + Hydra config composition + reproducibility) and
assert that each distinct unsafe knob trips the gate.

Everything is synthetic and CPU-only: no network, no model downloads, no real
spend. The Hydra compose check reads ``conf/`` from the repo, so these tests are
self-contained against the committed config tree.
"""

from guardian_research.agents.validate_proposal import validate_proposal
from guardian_research.common.budget import estimate_cost
from guardian_research.common.schemas import Proposal


def good_proposal() -> Proposal:
    """A known-good, fully synthetic proposal that should pass every check.

    Mirrors the shape an agent emits via ``ga propose``: small grid (2 lr x 2
    seeds = 4 jobs), allowed provider, synthetic data, complete scientific
    framing, a real git SHA, and a base_config that composes in Hydra.
    """
    return Proposal(
        name="safety_good_synthetic",
        experiment="arithmetic_catapult",
        hypothesis="H001: high-LR/cyclic schedules improve hard arithmetic accuracy.",
        expected_signal="non-baseline hard_acc exceeds baseline across seeds",
        metric="final_hard_acc",
        ablation="baseline_cosine control; vary lr x schedule x seed",
        stop_conditions=["stop a shard on NaN loss", "downgrade H001 if no crossover"],
        base_config="+exp=arithmetic_catapult model=tiny_transformer",
        sweep={"train.lr": [5.0e-4, 1.0e-3]},
        seeds=[0, 1],
        data_class="synthetic",
        provider="runpod",
        gpu="l40s",
        hours_per_job=0.25,
        per_job_cost_usd=estimate_cost(1, hours_per_job=0.25, gpu="l40s"),
        estimated_cost_usd=estimate_cost(4, hours_per_job=0.25, gpu="l40s"),  # 2 lr x 2 seeds
        max_cost_usd=25.0,
        reproducibility={"git_sha": "abc123def0"},
    )


def _failed(report) -> set[str]:
    return {c.name for c in report.checks if not c.passed}


# --------------------------------------------------------------------------- #
# The good case must pass — otherwise every negative test below is meaningless. #
# --------------------------------------------------------------------------- #
def test_good_synthetic_proposal_passes():
    report = validate_proposal(good_proposal())
    assert report.passed, [(c.name, c.detail) for c in report.checks if not c.passed]


# --------------------------------------------------------------------------- #
# Data class                                                                   #
# --------------------------------------------------------------------------- #
def test_private_data_class_fails():
    # Private persona data is OFF by default; it must trip the data-class gate
    # (and, since data_class == "private", the explicit-permission gate too).
    p = good_proposal()
    p.data_class = "private"
    report = validate_proposal(p)
    assert not report.passed
    failed = _failed(report)
    assert "data_class_allowed" in failed
    assert "private_data_permitted" in failed


# --------------------------------------------------------------------------- #
# Provider allow-list                                                          #
# --------------------------------------------------------------------------- #
def test_disallowed_provider_aws_fails():
    p = good_proposal()
    p.provider = "aws"  # not in the allowed_providers list
    report = validate_proposal(p)
    assert not report.passed
    assert "provider_allowed" in _failed(report)


# --------------------------------------------------------------------------- #
# Budget caps                                                                  #
# --------------------------------------------------------------------------- #
def test_per_job_over_5usd_cap_fails():
    # A single shard costing more than the $5 per-job policy cap must fail,
    # even if the total still fits the daily budget.
    p = good_proposal()
    p.per_job_cost_usd = 6.0  # > $5 per-job cap
    report = validate_proposal(p)
    assert not report.passed
    assert "per_job_within_policy" in _failed(report)


def test_total_over_25usd_daily_cap_fails():
    # A sweep total above the $25/day cap must fail the daily-budget gate.
    p = good_proposal()
    p.estimated_cost_usd = 50.0
    p.max_cost_usd = 50.0  # raise the self-declared ceiling so the daily cap is what bites
    report = validate_proposal(p)
    assert not report.passed
    assert "within_daily_budget" in _failed(report)


# --------------------------------------------------------------------------- #
# Scientific framing — a proposal is never "just try stuff".                   #
# --------------------------------------------------------------------------- #
def test_missing_hypothesis_fails():
    p = good_proposal()
    p.hypothesis = "   "  # whitespace only -> no real hypothesis
    report = validate_proposal(p)
    assert not report.passed
    assert "has_hypothesis" in _failed(report)


def test_missing_stop_conditions_fails():
    p = good_proposal()
    p.stop_conditions = []
    report = validate_proposal(p)
    assert not report.passed
    assert "has_stop_conditions" in _failed(report)


# --------------------------------------------------------------------------- #
# Config validity / reproducibility                                            #
# --------------------------------------------------------------------------- #
def test_base_config_that_does_not_compose_fails():
    # A base_config referencing a non-existent experiment must fail the Hydra
    # compose check (and therefore the whole proposal).
    p = good_proposal()
    p.base_config = "+exp=this_experiment_does_not_exist model=tiny_transformer"
    report = validate_proposal(p)
    assert not report.passed
    assert "config_composes" in _failed(report)


def test_missing_git_sha_fails():
    # Reproducibility requires a recorded, non-"unknown" git SHA.
    p = good_proposal()
    p.reproducibility = {"git_sha": "unknown"}
    report = validate_proposal(p)
    assert not report.passed
    assert "has_git_sha" in _failed(report)
