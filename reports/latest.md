# Repo state — living summary

_Living document. Update when tooling lands or a run is done. Reflects the repo as the
source of truth, not aspirations._

> **No empirical claims are made yet.** Everything below is *plumbing and toy-scale
> output*. Nothing here confirms or refutes any hypothesis in `planning/hypotheses/`.
> Where a number appears (e.g. the smoke run), it is tooling-validation output at a scale
> far too small and under-seeded to be evidence for or against a claim. The crossover /
> sample-efficiency / matched-compute tests that the hypotheses actually require have not
> been run.

## Doctrine (unchanged)

Local machine = control plane; cloud GPUs = disposable stateless workers (clone @ exact
SHA, run one declared job, upload results, die); repo = source of truth. No private /
persona data anywhere — synthetic or public only. Default code paths are CPU-only with no
model/dataset downloads.

## What is IMPLEMENTED tooling

The "spine" (24 tests + the CLI milestone sequence) and one full experiment vertical:

- **Artifacts / contracts:** `common/artifacts.py` (`RunWriter`, `new_run_id`),
  `common/schemas.py` (`RunResult`, `Proposal`, `ValidationReport`), `common/seeding.py`,
  `common/budget.py`, `common/env_info.py`, `common/hydra_utils.py`.
- **Config:** Hydra config tree under `conf/` — `config.yaml` plus groups
  `model/`, `schedule/`, `launcher/`, `tracking/`, `sweep/`, and `exp/` experiment files.
- **CLI commands** (auto-discovered, `commands/`): `train`, `analyze`, `launch`,
  `propose`, `validate-proposal`, `data`, `status`, `logs`, `cancel`, `collect`.
- **Tracking / reporting:** `tracking/mlflow_client.py` (no-ops if mlflow absent),
  `tracking/ingest.py`, `tracking/reports.py`.
- **Launchers:** `launchers/local.py` (in-process run), `launchers/skypilot.py`
  (cloud path — dry-run / budget-gated).
- **Agents:** `agents/propose.py` (drafts a next-sweep `Proposal` from prior runs),
  `agents/validate_proposal.py` (budget + data-class + reproducibility checks).
- **Arithmetic vertical (the one fully-built experiment — owns H001):**
  `experiments/arithmetic/{train,analyze}.py`, `data/arithmetic.py`
  (`CharTokenizer`, `build_splits`, easy/hard/OOD/carry-heavy splits),
  `models/tiny_transformer.py`, `schedules/schedules.py`
  (baseline cosine, one-cycle high-LR, cyclic LR, cyclic WD).

## What is STUBBED (honest stubs — interface reserved, no logic yet)

These are empty `__init__.py` placeholders; the runner keys exist in
`launchers/local.py` but the `run()` entry points are not implemented:

- **`experiments/dynamic_grokking/`** — owns **H002** (dynamic eval vs. static sampling
  at matched FLOPs). Runner key `dynamic_grokking` reserved; no harness yet.
- **`experiments/persona/`** — owns **H003** (active-question personalization). Runner key
  `persona` / entry `train_persona.run` reserved; no harness yet.
- **`experiments/cifar_robustness/`** — robustness micro-lab from the research program;
  no hypothesis file yet, no code yet.

## What has been RUN

- **CPU smoke / tooling-validation only.** `reports/runs/smoke_arithmetic.md` ingested
  **9** toy `arithmetic_catapult` runs (baseline_cosine, cyclic_lr, cyclic_weight_decay,
  onecycle_high_lr; seed 0; ~152k-param TinyTransformer), with figures under
  `reports/figures/arithmetic_catapult/`.
- **Crossover status:** Δ(best non-baseline − best baseline) on the hard split =
  **+0.000** — the baseline is **not yet** beaten. Per H001 this is **not a refutation**:
  it is an unfinished test (single seed, toy compute, well below what the claim needs).
  Several runs collapsed to 0 accuracy, consistent with toy-scale instability.
- **No GPU sweep, no matched-compute crossover study, no dynamic-eval run, no persona
  run** has been executed.

## Hypotheses (claims, not findings)

| ID | Claim (short) | Experiment | Status |
|----|---------------|------------|--------|
| H001 | Cyclic high-LR/WD schedules improve HARD arithmetic scaling ("curves cross") | `arithmetic_catapult` | tooling implemented; **not yet tested at adequate scale** |
| H002 | Dynamic eval beats static sampling at equal compute | `dynamic_grokking` | **stub**; harness not built |
| H003 | Active-question personalization > passive finetuning per query | `persona` | **stub**; harness not built |

See `planning/hypotheses/` for each claim's metric, expected signal, ablation, and
pre-registered STOP / kill conditions. See `planning/funding-demo-checklist.md` for the
three target crossovers (scientific / product / safety) — all currently **not-yet**.

## Standing reminders

- Distinguish *implemented tooling* from *empirical findings*. Toys may show null/negative
  results; that is allowed and is honestly reported as such.
- Dirty-tree runs are not reproducible (the smoke report flags some as dirty).
- Promote nothing to a "finding" without matched compute, multiple seeds, and the
  pre-registered stop conditions. Write a postmortem (`reports/postmortems/TEMPLATE.md`)
  at each decision point.
