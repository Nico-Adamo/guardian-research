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
- **Arithmetic vertical (owns H001):** `experiments/arithmetic/{train,analyze}.py`,
  `data/arithmetic.py` (`CharTokenizer`, `build_splits`, easy/hard/OOD/carry-heavy splits),
  `models/tiny_transformer.py` (now supports `pos_encoding = learned|none|rope` — rope/none
  enable OOD-length extrapolation), `schedules/schedules.py`
  (baseline cosine, one-cycle high-LR, cyclic LR, cyclic WD).
- **Modular-grokking testbed (also H001, cheap CPU):**
  `experiments/modular/{train,analyze}.py`, `data/modular.py` — `(a op b) mod p` with the
  memorize→grok metrics (`val_acc`, `grok_step`, `grok_gap`). Runner key `modular`,
  `+exp=arithmetic_modular_grok`.
- **Dynamic-grokking harness (owns H002):** `experiments/dynamic_grokking/run.py` +
  `ga grok` — runs a CPU toy comparing dynamic evaluation vs static sampling at equal
  compute. Real `full`/`last_layer` inner updates; `lora` target is an honest stub.
- **Persona pipeline (owns H003):** `experiments/persona/*` + `ga persona` — synthetic
  corpus, PRINCIPAL.md, evals, active questions, and a base/RAG/LoRA/dynamic-eval
  comparison. base + RAG-only are real; LoRA + dynamic-eval are honest stubs; judge is mock.

## What is STUBBED (honest stubs — flagged in code)

The *experiments* above run; these *parts within them* are intentionally not built:

- **`lora` inner-update target** in dynamic grokking (`full`/`last_layer` work).
- **LoRA & dynamic-eval persona variants** (gated behind the `[llm]` extra; reuse retrieval
  and record `is_stub=1`) and the **mock pairwise judge** (stylometric proxy, `judge_is_mock=1`).
- **Private→cloud encryption pipeline** (documented in `docs/security.md §4/§10`; the flag
  gates a path that does not exist — private data stays local).
- **`experiments/cifar_robustness/`** — robustness micro-lab; placeholder, no code/hypothesis yet.
- **Cloud result round-trip** — worker→home upload + a real `ga collect` (in progress).

## What has been RUN

- **CPU tooling-validation + two H001 probes only** (see
  `reports/postmortems/PM001-position-and-grokking-probes.md`):
  - `arithmetic_catapult` (base-10): 4 schedules at seed 0; `final_hard_acc` Δ = **+0.000**
    (baseline not beaten — unfinished test, not a refutation). Length-OOD accuracy is
    **0.000 for learned/none/rope** at CPU scale → reclassified as a GPU experiment.
  - `arithmetic_modular_grok` (`(a+b) mod 97`): the metric is strongly schedule-sensitive
    — `baseline_cosine` groks (`val_acc=1.0` @ step ~600); `cyclic_weight_decay` at its
    default 20× WD multiplier over-regularizes (`val_acc=0.028`). Report:
    `reports/runs/arithmetic_modular_grok.md`.
- **`ga grok` and `ga persona` toys** ran successfully (CPU); dynamic-eval edged static at
  toy scale and RAG-only lifted persona preference accuracy — both recorded honestly, not
  as claims.
- **No GPU sweep, no matched-compute multi-seed study, no cloud run** has been executed.
  (Cloud result round-trip is being closed in Phase 2.)

## Hypotheses (claims, not findings)

| ID | Claim (short) | Experiment | Status |
|----|---------------|------------|--------|
| H001 | Cyclic high-LR/WD schedules improve HARD arithmetic scaling ("curves cross") | `arithmetic_catapult` (GPU length-OOD) + `arithmetic_modular_grok` (CPU grokking) | toy testbeds run; metric movable on modular; **not yet tested at adequate scale/seeds** |
| H002 | Dynamic eval beats static sampling at equal compute | `dynamic_grokking` | CPU toy harness implemented; **not yet tested at scale** |
| H003 | Active-question personalization > passive finetuning per query | `persona` | CPU toy harness implemented (base/RAG real; LoRA/dyn-eval stubbed); **not yet tested** |

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
