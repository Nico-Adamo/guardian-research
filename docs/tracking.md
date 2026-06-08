# Experiment tracking

Tracking here is **local and self-hosted by default** — no managed SaaS — in
keeping with the security posture: a cloud worker is disposable and never carries
private data, so the authoritative record of a run must be a plain, ingestable
file on the control plane, not a row in someone else's database.

## Two layers, one source of truth

There are two tracking layers, and only one of them is authoritative.

1. **`runs/<experiment>/<run_id>/results.json` — the authoritative artifact.**
   Every run is written by `RunWriter`
   (`src/guardian_research/common/artifacts.py`) into a self-describing directory:

   ```
   runs/<experiment>/<run_id>/
     results.json     # the RunResult: params, final metrics, git+env provenance (SOURCE OF TRUTH)
     metrics.jsonl    # one JSON object per logged step (streamed during training)
     config.yaml      # the exact resolved Hydra config
     env.json         # git SHA / dirtiness + environment provenance
     artifacts/       # plots, checkpoint metadata, etc.
   ```

   Ingestion reads **only** `results.json` (and `metrics.jsonl` for curves). The
   whole portability contract is: *if you can see a run directory, you can ingest
   it* — laptop or cloud worker, identical code path.

2. **MLflow — a convenience mirror.** `src/guardian_research/tracking/mlflow_client.py`
   logs the same params/metrics/artifacts to a **local file store**
   (`file:./mlruns`) so you get a browsable UI. It is a *thin, non-fatal wrapper*:
   if MLflow is not installed or errors, every call becomes a no-op and the run
   still completes and still writes its `results.json`. **MLflow is never the
   source of truth** — losing `mlruns/` loses nothing reproducible.

## How runs log to MLflow

A runner opens a context manager and logs scalars + artifacts:

```python
from guardian_research.tracking.mlflow_client import start_run

with start_run(experiment, run_name, tags={}) as mlf:
    mlf.log_params({"lr": 1e-3, "schedule": "baseline_cosine"})
    mlf.log_metrics({"hard_acc": 0.31}, step=step)
    mlf.log_artifact(str(results_json_path))
```

- `log_params(dict)` — hyperparameters (coerced to scalars).
- `log_metrics(dict, step=int)` — time-series metrics keyed by step.
- `log_artifact(path)` — files (e.g. the run's `results.json` or a plot).

The tracking URI is always `file:<repo_root>/mlruns` (see
`tracking_uri()` / `mlruns_dir()`); nothing leaves the machine.

## Viewing the MLflow UI

```bash
make ui                     # → uv run mlflow ui --backend-store-uri file:./mlruns
# or directly:
uv run mlflow ui --backend-store-uri file:./mlruns
```

Then open the printed local URL. `mlruns/` is gitignored; it is a local cache you
can delete and rebuild at any time.

## Ingestion and reports (`ga analyze`)

Reports are generated **from `results.json`, not from MLflow.**

```bash
uv run ga analyze --experiment arithmetic_catapult --write reports/runs/arithmetic_catapult.md
# (Makefile target: `make report`)
```

What happens under the hood:

- `tracking/ingest.py:load_runs(experiment, since_days=...)` walks
  `runs/<experiment>/*/results.json` and loads each into a `RunResult`. It does
  **not** care where the directory came from.
- `tracking/reports.py:generate_experiment_report(...)` flattens runs into a
  comparison table, writes plots under `reports/figures/<experiment>/`, and emits
  a conservative markdown report — it describes *what was measured* and never
  claims a scientific result.

To verify what is ingestable without writing a report:

```bash
uv run ga collect            # lists every runs/<exp>/<id> results.json it can read
uv run ga collect <run_id>   # filter to one run
```

## How cloud workers get their results home

A disposable GPU worker writes the **same** `runs/<exp>/<id>/` directory locally
on the worker via `RunWriter`. Because the worker dies, results must be pulled
back to the control plane. The mechanism is an optional object store URI:

- Set `GUARDIAN_ARTIFACT_URI` (e.g. `s3://your-bucket/guardian-runs`) on the
  worker. The SkyPilot run block syncs `runs/` to that URI when it is non-empty
  (see `src/guardian_research/launchers/skypilot.py`). If it is empty, results
  simply die with the worker — nothing is silently retained.
- Pull them locally, then ingest exactly like a local run:

  ```bash
  aws s3 sync s3://your-bucket/guardian-runs/<run_id> runs/<exp>/<run_id>
  uv run ga collect            # verify it ingests
  uv run ga analyze --experiment <exp> --write reports/runs/<exp>.md
  ```

No private data flows through this path: the artifacts are synthetic-experiment
outputs (metrics, configs, plots), and the bucket is one the operator controls.

## Provenance & reproducibility

Every `results.json` records the **git SHA and tree dirtiness** plus the seed, so
any run can be tied back to exact code. Proposals (`ga propose`) carry the same
`git_sha` / `requires_exact_sha` fields, and cloud launches check out an exact
SHA — code travels by git, never by uploading a working tree.
