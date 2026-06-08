"""Structured result + proposal schemas.

These pydantic models are the contract between *everything*: training writes
``RunResult`` JSON, analysis/reporting reads it, agents emit ``Proposal`` YAML,
and the budget guard validates proposals into a ``ValidationReport``. Keeping
the schema explicit and versioned is what makes results portable from a
disposable cloud worker back to the local control plane.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0.0"

DataClass = Literal["public", "synthetic", "private"]


# --------------------------------------------------------------------------- #
# Run results                                                                  #
# --------------------------------------------------------------------------- #
class MetricPoint(BaseModel):
    step: int
    value: float


class GitInfo(BaseModel):
    sha: str
    dirty: bool
    branch: str | None = None


class EnvInfo(BaseModel):
    python: str
    torch: str | None = None
    platform: str
    hostname: str
    device: str = "cpu"
    cuda_available: bool = False


class RunResult(BaseModel):
    """The canonical artifact produced by every experiment run."""

    schema_version: str = SCHEMA_VERSION
    run_id: str
    experiment: str
    seed: int
    status: Literal["running", "completed", "failed"] = "running"
    created_at: str
    finished_at: str | None = None
    git: GitInfo
    env: EnvInfo
    # Flat, human-scannable hyperparameters (model size, lr, schedule, ...).
    params: dict[str, Any] = Field(default_factory=dict)
    # Full resolved Hydra config (for exact reproduction).
    config: dict[str, Any] = Field(default_factory=dict)
    # Summary scalars used for cross-run comparison and reports.
    final_metrics: dict[str, float] = Field(default_factory=dict)
    # Full per-metric history.
    metrics: dict[str, list[MetricPoint]] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    notes: str | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Sweep proposals (agent -> human-approval loop)                               #
# --------------------------------------------------------------------------- #
class Proposal(BaseModel):
    """A next-sweep proposal drafted by an agent from prior run metrics.

    A proposal is *data*, not an action. It must pass ``ga validate-proposal``
    (budget + data-class + config validity + reproducibility) and then explicit
    human approval before any money is spent.
    """

    schema_version: str = SCHEMA_VERSION
    name: str
    experiment: str
    # Scientific framing — all required so a proposal is never just "try stuff".
    hypothesis: str
    expected_signal: str
    metric: str
    ablation: str
    stop_conditions: list[str]
    # What to run.
    base_config: str  # e.g. "+exp=arithmetic_catapult model=tiny_transformer"
    sweep: dict[str, list[Any]] = Field(default_factory=dict)  # Hydra multirun axes
    seeds: list[int] = Field(default_factory=lambda: [0])
    # Safety / cost. A sweep is many *jobs* (shards). We track BOTH the per-job
    # cost (gated by the per-job policy cap) and the total sweep cost (gated by
    # the daily cap and by ``max_cost_usd`` = the budget the agent was given).
    data_class: DataClass = "synthetic"
    provider: str = "runpod"
    gpu: str = "l40s"
    hours_per_job: float = 0.25
    per_job_cost_usd: float = 0.0
    estimated_cost_usd: float = 0.0  # total across the whole sweep
    max_cost_usd: float = 25.0  # total budget ceiling for this sweep (from --budget-usd)
    # Provenance.
    created_from_runs: list[str] = Field(default_factory=list)
    reproducibility: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None

    def grid_size(self) -> int:
        n = 1
        for values in self.sweep.values():
            n *= max(1, len(values))
        return n * max(1, len(self.seeds))


# --------------------------------------------------------------------------- #
# Validation                                                                   #
# --------------------------------------------------------------------------- #
class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class ValidationReport(BaseModel):
    target: str
    passed: bool
    checks: list[CheckResult] = Field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> ValidationReport:
        self.checks.append(CheckResult(name=name, passed=passed, detail=detail))
        self.passed = self.passed and passed
        return self

    @classmethod
    def start(cls, target: str) -> ValidationReport:
        return cls(target=target, passed=True, checks=[])
