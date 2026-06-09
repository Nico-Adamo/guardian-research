---
name: new-experiment
description: Use to scaffold a new experiment / research lab in the guardian-research factory (a new runner + config + hypothesis + test), following the repo's spine contracts so it plugs into ga train / analyze / propose. Tier 0.
---

# new-experiment — scaffold a new lab the factory way

Add experiments so they conform to the spine and are immediately runnable with the
existing `ga` verbs. Keep it CPU-fast and download-free.

## Steps

1. **Hypothesis first.** Create `planning/hypotheses/H0NN-<slug>.md` with: claim,
   metric, expected signal, ablation/control, and explicit STOP/kill conditions.
   (Copy the structure of an existing H001–H003 file.)
2. **Runner module.** Create `src/guardian_research/experiments/<name>/run.py` (or
   `train.py`) exposing `run(cfg: dict) -> Path`. It MUST:
   - `seed_everything(cfg["seed"])`;
   - build data/model (reuse `models/tiny_transformer.py`, `data/*` where possible);
   - write results via `RunWriter` (`set_config`, `set_params`, `log_metrics`,
     `set_final`, `finish`) so `ga analyze` can ingest it;
   - log to MLflow via `start_run(...)` (it no-ops if mlflow is absent).
   Add an `__init__.py` to the package dir.
3. **Register the runner.** Add a key to `RUNNERS` in
   `src/guardian_research/launchers/local.py` mapping `"<name>" ->` the module path.
4. **Config.** Add `conf/exp/<name>.yaml` starting with `# @package _global_`,
   setting `experiment: <name>`, `runner: <name>`, and `data:` / `train:` toy
   blocks small enough to finish in seconds on CPU.
5. **Test.** Add `tests/test_<name>.py` that composes the config with tiny
   overrides, runs it, and asserts a valid `RunResult` (status completed, the key
   metrics present). Keep it fast.
6. **(Optional) CLI verb.** If it needs more than `ga train`, add
   `src/guardian_research/commands/<verb>.py` with `NAME`, `HELP`, `run(argv)`
   (auto-discovered). Use `compose_config`/`split_overrides` for Hydra-style args.
7. **Verify + record.** `uv run ruff check src tests && uv run pytest`, run the toy
   once (`uv run ga train +exp=<name>` or your verb), and update `reports/latest.md`.

Conventions: prefer boring, readable code; no private data; no scientific-success
claims in docs; small PR-sized commits on a branch.
