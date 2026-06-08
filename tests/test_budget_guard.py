"""Budget/safety guard tests — unsafe proposals MUST fail validation."""

from guardian_research.agents.validate_proposal import validate_proposal
from guardian_research.common.budget import BudgetGuard, estimate_cost
from guardian_research.common.schemas import Proposal


def good_proposal() -> Proposal:
    return Proposal(
        name="t",
        experiment="arithmetic_catapult",
        hypothesis="H001: high-LR/cyclic schedules improve hard arithmetic accuracy.",
        expected_signal="non-baseline hard_acc exceeds baseline across seeds",
        metric="final_hard_acc",
        ablation="baseline_cosine control; vary lr x wd x schedule",
        stop_conditions=["stop a shard on NaN loss", "downgrade H001 if no crossover"],
        base_config="+exp=arithmetic_catapult model=tiny_transformer",
        sweep={"train.lr": [5.0e-4, 1.0e-3]},
        seeds=[0, 1],
        data_class="synthetic",
        provider="runpod",
        gpu="l40s",
        hours_per_job=0.25,
        per_job_cost_usd=estimate_cost(1),
        estimated_cost_usd=estimate_cost(4),  # 2 lr x 2 seeds
        max_cost_usd=25.0,
        reproducibility={"git_sha": "abc123def0"},
    )


def test_good_proposal_passes():
    report = validate_proposal(good_proposal())
    assert report.passed, [(c.name, c.detail) for c in report.checks if not c.passed]


def test_per_job_over_cap_fails():
    p = good_proposal()
    p.per_job_cost_usd = 50.0  # > policy per-job cap of $5
    report = BudgetGuard().validate_proposal(p)
    assert not report.passed
    assert any(c.name == "per_job_within_policy" and not c.passed for c in report.checks)


def test_private_data_fails():
    p = good_proposal()
    p.data_class = "private"
    report = BudgetGuard().validate_proposal(p)
    assert not report.passed


def test_disallowed_provider_fails():
    p = good_proposal()
    p.provider = "aws"
    report = BudgetGuard().validate_proposal(p)
    assert not report.passed
    assert any(c.name == "provider_allowed" and not c.passed for c in report.checks)


def test_total_over_daily_cap_fails():
    p = good_proposal()
    p.estimated_cost_usd = 999.0
    p.max_cost_usd = 999.0
    report = BudgetGuard().validate_proposal(p)
    assert not report.passed
    assert any(c.name == "within_daily_budget" and not c.passed for c in report.checks)


def test_missing_scientific_framing_fails():
    p = good_proposal()
    p.hypothesis = ""
    p.stop_conditions = []
    report = validate_proposal(p)
    assert not report.passed


def test_preflight_requires_clean_tree_and_sha():
    report = BudgetGuard().preflight_launch(
        provider="runpod",
        data_class="synthetic",
        per_job_cost_usd=0.21,
        total_cost_usd=2.0,
        max_total_cost_usd=5.0,
        git_dirty=True,  # dirty tree -> must fail
        git_sha="unknown",  # no SHA -> must fail
        dry_run_done=False,  # no dry run -> must fail
    )
    assert not report.passed
    names = {c.name for c in report.checks if not c.passed}
    assert "clean_git_tree" in names
    assert "exact_commit_sha" in names
    assert "dry_run_first" in names
