# guardian-research

An **experiment factory** for investigating three coupled ideas from Gwern
Branwen's essays *“Human-like Neural Nets by Catapulting”* and *“Guardian
Angels: LLM Personalization for Productivity and Security”*:

1. **Catapult / grokking training dynamics** — can high-learning-rate, cyclical
   schedules push small overparameterized models out of brittle memorizing
   basins into flatter, more *algorithmic* basins that generalize on *hard*
   out-of-distribution examples?
2. **Dynamic grokking** — can a handful of test-time weight updates ("pondering")
   on a single hard problem buy something that static repeated sampling cannot?
3. **Guardian-Angel personalization** — can in-weight personalization (dynamic
   evaluation + active elicitation) make a model materially better at one
   *principal's* taste than a prompt-only or RAG-only baseline, while being
   harder to push around?

> **This repo is tooling, not a result.** Everything below describes
> *implemented machinery* for running and logging these experiments cheaply and
> reproducibly. It makes **no claim** that catapulting works, that dynamic
> grokking helps, or that personalization is safe. Toys here are explicitly
> allowed to produce null or negative results, and reporting them honestly is a
> goal, not a failure. See `planning/guardian/research-program.md` for the
> falsification-first framing and the pre-registered kill criteria.

**Design in one line:** *local machine = control plane, cloud GPUs = disposable
stateless workers, the repo = source of truth.*

---

## Quickstart

The project uses [`uv`](https://docs.astral.sh/uv/) for environment management
and Python 3.11. `just` is **optional** — every recipe is mirrored exactly in a
portable `Makefile`, so use whichever you have.

```bash
uv sync                 # create the environment (CPU-only, NO model downloads)

just setup              # == uv sync                 (Makefile: make setup)
just test               # fast CPU test suite        (make test)
just smoke              # CPU-only toy training + report (make smoke)
```

- `uv sync` installs only the lean core (PyTorch CPU, Hydra, MLflow, etc.). No
  Hugging Face downloads, no network access, no GPU required.
- `just test` / `make test` runs `pytest -q -m "not gpu and not slow"`.
- `just smoke` / `make smoke` trains a tiny transformer on synthetic arithmetic
  for a few hundred CPU steps and writes a markdown report. It is the canonical
  "does my checkout work?" check.

If you do not have `just`, run `make help` to list the mirrored targets.

---

## The milestone command sequence

This is the end-to-end "control-plane loop" the repo is built around. Every step
below is CPU-only and spends **no money** — the cloud step is a *dry run* that
only renders YAML and commands. The `ga` CLI is installed by `uv sync`
(`uv run ga ...`).

```bash
# 1) Train a baseline run (standard cosine schedule).
uv run ga train +exp=arithmetic_catapult model=tiny_transformer \
    schedule=baseline_cosine seed=0

# 2) Train a catapult-candidate run (one-cycle high-LR schedule).
uv run ga train +exp=arithmetic_catapult model=tiny_transformer \
    schedule=onecycle_high_lr seed=0

# 3) Analyze: collect local runs into a comparison report.
uv run ga analyze --experiment arithmetic_catapult \
    --write reports/runs/arithmetic_catapult.md

# 4) Dry-run a cloud sweep (renders the worker spec + cost estimate; spends $0).
uv run ga launch --dry-run --provider skypilot \
    +exp=arithmetic_catapult sweep=arith_lr_wd_seed_v0

# 5) Propose the next sweep from prior runs (drafts a YAML; does NOT launch).
uv run ga propose --experiment arithmetic_catapult --budget-usd 25 \
    --write reports/proposals/next_arith_sweep.yaml

# 6) Validate the proposal (budget + data-class + config-composes + repro checks).
uv run ga validate-proposal reports/proposals/next_arith_sweep.yaml
```

What each step does:

| Step | Command | What it does | Spends money? |
|------|---------|--------------|---------------|
| 1 | `ga train ... schedule=baseline_cosine` | Trains one experiment **in-process** from a composed Hydra config; writes `runs/<exp>/<run_id>/{results.json,metrics.jsonl,config.yaml,env.json}` and logs to local MLflow. The **control** arm. | No (local CPU) |
| 2 | `ga train ... schedule=onecycle_high_lr` | Same, with the high-LR/cyclical schedule — the catapult **candidate** arm. | No (local CPU) |
| 3 | `ga analyze --experiment ...` | Ingests all local runs for an experiment, builds a comparison table + figures, and writes a markdown report under `reports/runs/`. | No |
| 4 | `ga launch --dry-run ...` | Renders the exact SkyPilot worker YAML (clone-at-SHA, `uv sync`, run shards, die), expands the sweep into explicit `ga train` commands, and prints a **cost estimate vs. the policy caps**. Launches nothing. | No |
| 5 | `ga propose --experiment ... --budget-usd N` | An analyzer agent drafts a next-sweep `Proposal` YAML (hypothesis, axes, expected signal, ablation, stop conditions, cost) from prior runs. | No |
| 6 | `ga validate-proposal PATH` | Runs the proposal through the **budget guard** + a Hydra-compose check + reproducibility checks (git SHA present, cost matches grid). Prints PASS/FAIL. A FAIL blocks any launch. | No |

Steps 5→6 are the **propose → validate → approve → launch** loop, closed by two
more verbs:

```bash
ga approve reports/proposals/next_arith_sweep.yaml --by <you>   # records human sign-off
ga launch --dry-run --proposal reports/proposals/next_arith_sweep.yaml   # then add --yes
```

`ga approve` binds the sign-off to the proposal's content hash **and** the current
commit (edit the proposal or move HEAD → approval is invalidated, must re-approve).
`ga launch --proposal …` refuses a real launch unless that approval is valid, on top
of the budget preflight. Only a human, after a passing validation and approval, may
run `ga launch --proposal … --yes` to spend real money.

When a worker finishes it uploads `runs/` to `$GUARDIAN_ARTIFACT_URI/<git_sha>/runs/`;
`ga collect --from <uri> --sha <sha>` pulls them home and ingests them — that closes
the round-trip so cloud results reach the local control plane.

The other CLI verbs round out the loop: `ga data` (generate synthetic data),
`ga status` (budget ledger + run summary), `ga logs <run_id>`, and `ga cancel <run_id>`.

---

## The three research tracks (and where they live)

The repo is organized around the synthesis in
`planning/guardian/research-program.md`: a two-timescale system whose **slow
weights** favor cleaner abstractions (catapult) and whose **fast weights** adapt
online to a principal (Guardian). The tracks are deliberately falsifiable; each
has a hypothesis note under `planning/hypotheses/` where present.

### Track 1 — Catapult / grokking (slow weights, "better abstraction")
*Does a high-LR/cyclical recipe eventually **cross over** and beat a standard
recipe on hard extrapolation at matched compute?*

- Experiments: `src/guardian_research/experiments/arithmetic/` (synthetic
  addition/etc. with a *hard* held-out split) and the
  `cifar_robustness/` placeholder for the small-image robustness micro-lab.
- Configs: `conf/exp/arithmetic_catapult.yaml`, schedules under
  `conf/schedule/` (`baseline_cosine`, `onecycle_high_lr`, `cyclic_lr`,
  `cyclic_weight_decay`).
- Building blocks: `models/tiny_transformer.py`, `data/arithmetic.py`,
  `schedules/schedules.py`.
- The success condition is a **crossover on the hard split**, not lower average
  loss. If no crossover ever appears at toy scale, the thesis is downgraded.

### Track 2 — Dynamic grokking (test-time "pondering")
*Do a few **online weight updates** on one hard problem beat the same FLOPs
spent on repeated sampling?*

- Experiment runner: `src/guardian_research/experiments/dynamic_grokking/`
  (runner key `dynamic_grokking`, exposing `run(cfg)`).
- This is the highest-novelty / easiest-to-overstate track; treat its outputs
  with the most skepticism.

### Track 3 — Guardian-Angel personalization (fast weights, "better identity")
*Can in-weight personalization beat prompt-only / RAG-only baselines for one
principal — and is the personalized model harder to jailbreak?*

- Experiment runner: `src/guardian_research/experiments/persona/`
  (runner key `persona`, `train_persona.py` exposing `run(cfg)`).
- **Synthetic personas only.** No real/private persona data is ever used,
  logged, or uploaded by any default code path. See
  [Data handling](#data-handling) and `docs/security.md`.

Source ideas live in `planning/guardian/` (the two essays plus the synthesis and
deep-research notes). Read those for *intent*; read the code for *contracts*.

---

## Repo layout tour

```
guardian-research/
  README.md                 # you are here
  pyproject.toml            # deps + [tool.guardian] budget/autonomy policy
  Makefile / justfile       # mirrored command recipes (just is optional)
  .env.example              # secrets/config template — copy to .env, never commit

  planning/
    guardian/               # source essays + research-program synthesis
    hypotheses/             # falsifiable claims (H001 catapult, H002 dynamic-eval, ...)
    decisions/              # Architecture Decision Records (ADRs)

  conf/                     # Hydra config groups
    config.yaml             # root: model / schedule / launcher / tracking / sweep
    exp/                    # +exp=<name> experiment definitions (# @package _global_)
    model/  schedule/  launcher/  tracking/  sweep/

  src/guardian_research/
    cli.py                  # `ga` — auto-discovers commands/
    commands/               # one module per CLI verb (NAME, HELP, run(argv)->int)
    common/                 # artifacts (RunWriter), seeding, budget, schemas, paths
    launchers/              # local.py (in-process) + skypilot.py (dry-run/gated)
    tracking/               # mlflow_client, ingest, reports
    data/                   # arithmetic.py (synthetic data + CharTokenizer)
    models/                 # tiny_transformer.py (CPU-friendly GPT)
    schedules/              # LR / weight-decay schedules
    experiments/            # arithmetic/, cifar_robustness/, dynamic_grokking/, persona/
    agents/                 # propose / validate_proposal (the autonomy loop)

  infra/                    # Docker / SkyPilot task specs / local MLflow compose
  data/                     # synthetic + public data only (DVC-compatible layout)
  runs/                     # gitignored local run cache (results/metrics/artifacts)
  artifacts/                # gitignored local artifact cache
  mlruns/                   # gitignored local MLflow file store
  reports/                  # runs/ proposals/ figures/ postmortems/
  tests/                    # CPU, non-slow by default
```

Each `commands/` module is **auto-discovered**: define `NAME`, `HELP`, and
`run(argv) -> int` and it shows up in `ga --help` with no edits to `cli.py`.
Each experiment runner exposes `run(cfg: dict) -> Path` and is mapped from a
`runner` key in `launchers/local.py`.

---

## Local dev vs. cloud launches

The whole point of the control-plane design (see
[ADR-0001](planning/decisions/ADR-0001-experiment-factory.md) and
[ADR-0002](planning/decisions/ADR-0002-disposable-gpu-workers.md)):

- **Develop locally.** Edit code, run CPU smoke tests, inspect runs, review agent
  proposals, and maintain the research ledger on your machine. *Never SSH into a
  GPU box and edit files there.*
- **Execute on cattle, not pets.** A cloud worker `git clone`s the repo,
  `git checkout`s an **exact SHA**, `uv sync`s, runs one declared
  experiment/sweep shard, uploads results to an object store, and shuts itself
  down. It receives **code via git@SHA only** — no `workdir` upload, no local
  files, no private data.
- **Default-safe launching.** `ga launch` defaults to `--dry-run`. A real launch
  requires *all* of: the `[cloud]` extra installed, `GUARDIAN_ALLOW_REAL_LAUNCH=1`
  set, a clean git tree at a known SHA, a prior dry-run, a passing budget
  preflight, and an explicit `--yes`.

See `docs/setup.md` for the optional `[cloud]` / `[llm]` / `[dvc]` extras.

---

## Tracking

Tracking is **local and self-hosted by default** — no SaaS, nothing leaves your
machine.

```bash
just ui     # or: make ui
            # == uv run mlflow ui --backend-store-uri file:./mlruns
```

- Every `ga train` writes a self-contained run directory under `runs/<exp>/...`
  (`results.json`, `metrics.jsonl`, `config.yaml`, `env.json`, `artifacts/`) via
  the `RunWriter` in `common/artifacts.py`, and *also* logs params/metrics/
  artifacts to a local MLflow **file store** at `file:./mlruns`.
- MLflow is imported lazily and **degrades to a no-op** if it is missing, so the
  core never hard-depends on it.
- For live cross-machine tracking you would point `MLFLOW_TRACKING_URI` at a
  small always-on endpoint (not your laptop) — but that is optional and off by
  default.

## Data handling

- **Synthetic and public data only.** The arithmetic track generates its data
  on the fly (`data/arithmetic.py`); personas are synthetic. The budget policy
  (`pyproject.toml` `[tool.guardian]`) restricts `allowed_data_classes` to
  `["public", "synthetic"]` and sets `allow_private_persona_data = false`.
- **No private persona data, ever, anywhere** — not in tests, not in toys, not
  on a worker. The private data class is gated behind an explicit policy flag
  that defaults to off and would additionally require an encryption + approval
  plan that does not yet exist. See `docs/security.md`.
- **DVC-compatible, DVC-optional.** `data/` uses a `raw/processed/splits` layout
  that DVC can version, but DVC is an optional extra (`uv sync --extra dvc`); the
  repo works fine without it. Large/generated assets stay out of git.

---

## Safety gates and autonomy tiers

Spending and data access are governed by a single source of truth — the
`[tool.guardian]` table in `pyproject.toml` — enforced by the budget guard in
`common/budget.py`. Environment variables can only **tighten** the caps, never
loosen them. The current policy caps a single job at **$5**, a day at **$25**,
and the whole program at **$2,000**, on providers `runpod`/`lambda`/`modal`,
with data classes restricted to `public`/`synthetic`.

Three autonomy tiers govern what an automated agent may do on its own:

- **Tier 0 — unrestricted (no money).** Docs, tests, CPU runs, summarizing
  results, and *drafting* proposals (`ga propose`). Fully autonomous.
- **Tier 1 — bounded automation.** Small, budget-capped GPU jobs — allowed only
  if **every** gate passes: per-job ≤ cap, daily ≤ cap, allowed provider,
  allowed data class, clean git tree, exact commit SHA, and a prior `--dry-run`.
- **Tier 2 — approval required.** Anything larger, anything touching private
  data, any upload of sensitive data, or any destructive action (deleting
  artifacts, changing infra). Requires a human.

Before any real (money-spending) launch, `BudgetGuard.preflight_launch` runs a
hard gate covering provider, data class, per-job cost, total cost, daily budget,
clean tree, exact SHA, and dry-run-first. Untrusted external content must never
flow directly into persistent persona weights without quarantine + replay +
review.

**Full policy, threat model, and the safe operating procedure for autonomous
agents:** see [`docs/security.md`](docs/security.md).

---

## Where to go next

- New here? `docs/setup.md` — detailed local setup, extras, and troubleshooting.
- Why this architecture? `planning/decisions/ADR-0001-experiment-factory.md`.
- How do cloud workers stay safe? `planning/decisions/ADR-0002-disposable-gpu-workers.md`.
- What are we actually trying to falsify? `planning/guardian/research-program.md`
  and `planning/hypotheses/`.
