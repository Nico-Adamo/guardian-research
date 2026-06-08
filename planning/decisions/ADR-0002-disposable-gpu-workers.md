# ADR-0002: Disposable, stateless GPU workers (clone @SHA, run, die)

- **Status:** Accepted
- **Date:** 2026-06-08
- **Deciders:** Guardian Research
- **Related:** [ADR-0001 — experiment factory / local control plane](ADR-0001-experiment-factory.md),
  `src/guardian_research/launchers/skypilot.py`, `src/guardian_research/common/budget.py`,
  `docs/security.md`

## Context

[ADR-0001](ADR-0001-experiment-factory.md) commits to a local control plane with
cloud GPUs as *disposable workers*. This ADR fixes the **worker contract**: how
code and configuration reach a GPU box, what data it is allowed to see, and how a
job goes from idea to actually spending money.

The constraints are sharp:

- **Reproducibility.** Every result must trace to an exact commit. A worker
  running uncommitted or drifted code is worthless for a logging-heavy program.
- **No data exfiltration.** The Guardian-Angel track will eventually involve the
  most sensitive possible data (a principal's corpus). Even in the synthetic-only
  prototype phase, the *machinery* must make it structurally impossible to ship
  local or private files to a third-party cloud worker. "Don't upload the wrong
  folder" cannot be a matter of remembering.
- **Bounded spend.** The total budget is ~$2,000. There must be a hard gate
  between an agent (or a tired human) wanting to launch and money actually being
  spent, with per-job, daily, and total ceilings.
- **No silent burn.** An idle GPU is wasted budget; a worker that forgets to shut
  down is a slow leak.

A naive launcher would use a tool's "sync my working directory to the box"
feature (e.g. SkyPilot `workdir` / `file_mounts`). That is convenient and
exactly wrong here: it would happily ship whatever is in the local tree —
including `.env`, scratch data, or (later) private persona corpora — to a rented
machine governed by the third-party doctrine.

## Decision

**Cloud workers are stateless cattle that receive code via git@SHA only, never
local files, and die when done.** Concretely:

1. **Code travels by git, at an exact SHA.** The rendered worker spec does
   `git clone "$GUARDIAN_REPO_URL" && git checkout "$GIT_SHA" && uv sync`. There
   is **no `workdir:` upload and no `file_mounts:` of local paths** — this is a
   deliberate, documented refusal in `launchers/skypilot.py`. Local/private data
   therefore *cannot* leak onto a worker, because nothing local is shipped.
2. **One worker = one declared shard.** A worker runs the explicit `ga train`
   commands for its experiment/sweep shard, then exits. Sweeps are expanded into
   explicit per-shard commands at render time so the worker is fully declarative.
3. **Results are uploaded, then the worker dies.** If `GUARDIAN_ARTIFACT_URI` is
   set, the worker syncs `runs/` to that object store; otherwise the results die
   with the worker (the control plane is expected to collect them). Autostop /
   `--down` is part of the worker lifecycle so an idle box never lingers.
4. **`ga launch` is default-safe.** It defaults to `--dry-run`, which only
   *renders* the worker YAML, expands the shard commands, and prints a cost
   estimate against the policy caps. A dry run launches nothing and spends $0.
5. **A real launch must pass a hard preflight + explicit opt-ins.** Spending
   money requires *all* of: the `[cloud]` extra installed,
   `GUARDIAN_ALLOW_REAL_LAUNCH=1`, a prior dry-run, an explicit `--yes`, and a
   passing `BudgetGuard.preflight_launch` covering provider, data class, per-job
   cost, total cost, daily budget, **clean git tree**, **exact commit SHA**, and
   **dry-run-first**.
6. **The propose → validate → approve → launch loop is the path to spend.** An
   analyzer agent drafts a `Proposal` (`ga propose`); `ga validate-proposal` runs
   it through the budget/data-class checks plus a Hydra-compose and
   reproducibility check; a human approves; only then may the gated launch run.
   Untrusted external content must never flow directly into persistent persona
   weights — it is quarantined, replayed, and reviewed first (see
   `docs/security.md`).

Secrets (repo URL, provider keys) are read only from the environment or a local
`.env`, never committed, and the only thing a worker authenticates with is a
**read-only** repo URL plus whatever the provider needs to bill — not the
control plane's full credentials.

## Consequences

### Positive

- **Exfiltration is structurally prevented, not merely discouraged.** Because no
  local files are ever shipped, there is no folder to accidentally include. This
  is the single most important property for the eventual personalization work.
- **Every run is reproducible.** A result is `git SHA + composed config + seed`;
  a worker cannot run anything else.
- **Spend is bounded and auditable.** The preflight is a hard wall; the local
  ledger records intent; dry-run-first means no surprise bills.
- **No pets, no silent burn.** Workers are interchangeable and self-terminating;
  losing one loses nothing important.
- **Agents can drive the boring 95%.** Drafting and validating proposals is
  Tier-0 (free); only the final launch needs a human, which keeps automation
  useful without making it dangerous.

### Negative / costs

- **A commit is required before every cloud run.** You cannot test an
  uncommitted tweak on a GPU; you must commit (clean tree) first. This is a
  feature for reproducibility but a friction for quick experiments. Mitigated by
  doing all quick iteration locally on CPU.
- **Results must be deliberately collected.** Since workers die, anything not
  uploaded is gone. The `GUARDIAN_ARTIFACT_URI` convention and `ga collect`
  exist to routinize this, but it is an extra moving part.
- **Re-`uv sync` per worker.** Each worker rebuilds its environment from
  scratch, costing a little startup time/compute. Acceptable for our shard sizes;
  a prebuilt image (`infra/docker/`) is the optimization if it ever bites.
- **The dry-run-first / `--yes` discipline can feel heavy** for a trivial launch.
  We accept the friction as the price of an irreversible-action gate.

### Neutral / follow-ups

- The provider list (`runpod`/`lambda`/`modal`) and cost model live in
  `[tool.guardian]` and `common/budget.py`; tightening is allowed via env vars,
  loosening is not.
- This ADR governs *mechanism*. The threat model, data-class policy, and the
  full safe-operating procedure for autonomous agents are documented in
  `docs/security.md`.
