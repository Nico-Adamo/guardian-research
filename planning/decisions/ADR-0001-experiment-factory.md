# ADR-0001: An experiment factory with a local control plane

- **Status:** Accepted
- **Date:** 2026-06-08
- **Deciders:** Guardian Research
- **Related:** [ADR-0002 — disposable GPU workers](ADR-0002-disposable-gpu-workers.md),
  `planning/guardian/planning.md`, `planning/guardian/research-program.md`

## Context

The research program (see `planning/guardian/research-program.md`) is a
dual-track effort spanning catapult/grokking training dynamics, dynamic
grokking, and Guardian-Angel personalization. Concretely that means **many
carefully logged variations**: schedule × learning-rate × weight-decay × seed ×
data-filter sweeps for the catapult track, and append-only logs + dynamic
evaluation + active elicitation loops for the personalization track. The total
compute budget is small (~$2,000), so iteration speed and *not wasting runs*
matter more than peak FLOPs.

Two obvious shapes present themselves:

1. **A single long-lived cloud VM.** Spin up one GPU box, SSH in, edit code in
   place, run experiments, keep results on the box, and treat it as the project's
   home.
2. **An experiment factory with a local control plane.** Develop and orchestrate
   on a local machine; treat cloud GPUs as disposable workers that pull an exact
   commit, run one declared job, upload results, and die; keep the repo as the
   single source of truth for all hypotheses, configs, data-generation code,
   metrics, and reports.

The single-VM shape is tempting because it is the fastest way to "just run
something on a GPU." But it has well-known failure modes for a research program
that lives or dies on reproducibility and careful logging:

- **State drifts onto the box.** Code edited in place diverges from git;
  "which exact code produced this number?" becomes unanswerable. Results,
  checkpoints, and ad-hoc scripts accumulate only on the VM.
- **The VM is a pet.** It must be kept alive, patched, and paid for even while
  idle. Losing it (eviction, crash, an expired card) loses the work.
- **Cost has no natural ceiling.** An always-on GPU silently burns budget; there
  is no structural gate between "I had an idea" and "I spent money."
- **It does not scale to sweeps.** Catapult work needs dozens of small shards in
  parallel; one VM serializes them or requires manual fan-out.
- **It is hostile to autonomy and to safety.** An agent operating on a
  long-lived box with full credentials and mutable state is hard to bound; the
  personalization track will eventually involve sensitive data where casual
  cloud state is exactly wrong.

The local-control-plane shape directly answers the workflow questions raised in
`planning/guardian/planning.md`: develop locally, execute on cattle, keep
tracking local, and let agents *propose* freely but *spend* only through a gate.

## Decision

**Build this as an experiment factory, not a single cloud VM project.** The
roles are:

- **Local machine = control plane.** Code editing, CPU smoke tests, result
  inspection, report generation, proposal review, and the research ledger all
  live and run locally. The `ga` CLI is the single command surface; new
  capabilities register as auto-discovered command modules rather than ad-hoc
  scripts.
- **Cloud GPUs = disposable workers.** A worker clones the repo at an **exact
  git SHA**, `uv sync`s, runs one declared experiment or sweep shard, uploads
  logs/artifacts/results, and shuts down. Workers are cattle, never pets. (The
  worker contract is the subject of [ADR-0002](ADR-0002-disposable-gpu-workers.md).)
- **Repo = source of truth.** Hypotheses (`planning/hypotheses/`), Hydra configs
  (`conf/`), launcher specs (`launchers/`, `infra/`), data-generation code
  (`data/`), the budget/autonomy policy (`[tool.guardian]` in `pyproject.toml`),
  metrics/reports (`reports/`), and ADRs all live in git. No important state
  exists only on a VM.
- **Tracking = local by default.** Runs write self-contained directories under
  `runs/` and log to a local MLflow file store (`file:./mlruns`). No SaaS is
  required; a remote tracking endpoint is optional and would be a small
  always-on service, never the laptop.
- **Autonomy through a proposal/approval loop.** Agents may run Tier-0 work
  (docs, tests, CPU runs, summarizing, drafting proposals) freely. Anything that
  spends money flows through `results → propose → validate → human approve →
  launch`, with a hard budget preflight before any real launch.

This is implemented today: `ga train`/`analyze` (local control plane),
`ga launch --dry-run` (default-safe rendering), and `ga propose` /
`ga validate-proposal` (the autonomy loop), all enforced by `common/budget.py`.

## Consequences

### Positive

- **Reproducibility by construction.** Every number traces to a git SHA + a
  composed config + a seed, because workers can only run committed code.
- **Cost is gated, not hoped-for.** There is a structural wall between proposing
  and spending; idle workers cannot exist (they die), and a preflight enforces
  per-job/daily/total caps.
- **Sweeps are natural.** A sweep is just N stateless shards; fan-out is a
  property of the worker model, not manual SSH labor.
- **Safe to automate incrementally.** The tier model lets agents do the boring,
  zero-cost majority of the work autonomously while keeping money and private
  data behind a human gate.
- **Cheap iteration is favored over glamorous runs.** The design pushes most
  effort to fast CPU/L40S-class loops, which matches the budget logic in the
  research program.

### Negative / costs

- **More upfront scaffolding** than `ssh` + a training script: a CLI, config
  composition, an artifact writer, a launcher spec, and a budget guard had to be
  built before the first GPU run.
- **A round-trip discipline.** "Edit → commit → launch → collect" is slower for a
  one-off experiment than editing on a live box. We accept this; the loss of
  reproducibility from in-place editing is not worth the convenience.
- **Results must be actively collected.** Because workers die, anything not
  uploaded is lost. `ga collect` and an artifact-store convention exist to make
  this routine, but it is a step that a single-VM workflow would not need.
- **Local tracking needs a sync story** for genuinely cross-machine live views;
  we defer that to an optional remote endpoint rather than baking it in.

### Neutral / follow-ups

- The exact worker contract (clone-at-SHA, no local data, upload-and-die) is
  specified separately in [ADR-0002](ADR-0002-disposable-gpu-workers.md).
- This ADR commits to the *shape*, not to any scientific claim. The factory is
  tooling; whether catapulting or dynamic grokking or personalization *works* is
  an empirical question the factory exists to test honestly, including null
  results.
