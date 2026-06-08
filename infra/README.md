# infra/ — images, cloud tasks, and tracking

This directory holds the *cloud/infra surface* of guardian-research. None of it
is required for the default CPU deliverable: `uv sync` + `uv run ga ...` runs
everything locally with no network or model downloads. The infra here exists so
that, when you choose to, you can run on a real accelerator and keep a shared
tracking UI — without ever changing the safety model.

Layout:

```
infra/
  docker/
    Dockerfile.cpu        # hermetic CPU image: runs tests / smoke trains
    Dockerfile.gpu        # CUDA 12.4 image: entrypoint runs `ga train`
  sky/
    train.yaml            # ONE run on a disposable worker
    sweep.yaml            # managed-jobs sweep (budget-gated)
    eval.yaml             # pull artifacts + `ga analyze`
  mlflow/
    docker-compose.yml    # self-hosted tracking server (sqlite + local artifacts)
```

## Design doctrine (why infra looks like this)

- **Local control plane, disposable GPU workers.** Your machine decides *what*
  to run; cloud workers are cattle that clone the repo at an exact SHA, run one
  declared thing, and die.
- **Repo is the source of truth.** Code travels by git@SHA, never by uploading
  your working tree.
- **No private data, ever.** Synthetic data only; nothing private is uploaded
  to a worker or logged to tracking.

## Building images

From the repo root:

```bash
# CPU image — default CMD runs the test suite (CPU-only, no downloads).
docker build -f infra/docker/Dockerfile.cpu -t guardian-cpu .
docker run --rm guardian-cpu                       # run tests
docker run --rm guardian-cpu \
  uv run ga train +exp=arithmetic_catapult train.max_steps=50 device=cpu

# GPU image — entrypoint is `ga train`; args are Hydra overrides.
docker build -f infra/docker/Dockerfile.gpu -t guardian-gpu .
docker run --rm --gpus all guardian-gpu +exp=arithmetic_catapult_gpu seed=0
```

CUDA torch wheels: the core lockfile pins a portable `torch`. To get a
CUDA-matched wheel for the GPU base, install from the PyTorch CUDA index inside
the image (documented in `Dockerfile.gpu`):

```bash
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch
```

We leave that as a documented manual step so the core lockfile stays
CPU-portable and `uv sync` works on any machine.

## How launches work

The blessed path is `ga launch`, which renders the exact SkyPilot YAML and runs
the budget/safety preflight before anything is submitted:

```bash
# 1) Always dry-run first (spends nothing, just renders YAML + cost):
uv run ga launch --dry-run +exp=arithmetic_catapult_gpu seed=0

# 2) A REAL launch is gated: it needs --yes AND a passing preflight
#    (per-job cap, daily cap, allowed provider, allowed data class,
#     clean git tree, exact commit SHA, and a prior dry-run). Even then,
#    actual submission only happens with GUARDIAN_ALLOW_REAL_LAUNCH=1 and the
#    [cloud] extra installed; otherwise the exact `sky launch` command is
#    printed for a human to run.
```

The YAML files in `sky/` mirror exactly what
`src/guardian_research/launchers/skypilot.py` renders. They are the
hand-inspectable reference and a fallback for driving `sky` directly:

```bash
GUARDIAN_REPO_URL=https://github.com/you/guardian-research.git \
GIT_SHA=<exact-sha> \
sky launch -i 5 --down -y infra/sky/train.yaml
```

Each task's lifecycle is the same: **clone @SHA → `uv sync` → run one declared
experiment/sweep → (optionally) upload results → die.**

## The no-data-upload invariant

This is the single most important property of the infra, and it is enforced
structurally, not by convention:

- The sky tasks have **no `workdir:`** and **no `file_mounts:`** of local paths.
  SkyPilot's workdir upload — which would ship your local working tree to a
  worker — is deliberately not used.
- Code reaches the worker only via `git clone $GUARDIAN_REPO_URL` +
  `git checkout $GIT_SHA`. If it is not committed and pushed, it does not run.
- Results leave the worker only via an explicit object store you configure
  (`GUARDIAN_ARTIFACT_URI`); if unset, results die with the worker.
- Secrets are read from the environment / a local `.env` (gitignored), never
  baked into images and never committed. See the repo `.env.example`.

Net effect: there is no code path by which local or private/persona data
reaches a cloud worker.

## autostop / tearing workers down

Workers are billed by the second, so an idle worker is wasted budget. Always
launch with autostop and/or `--down`:

- `sky launch -i <min> --down ...` — stop after `<min>` idle minutes, then tear
  the cluster down entirely.
- `sky jobs launch --down ...` — managed jobs (used for sweeps) queue,
  auto-recover on spot preemption, and tear down on completion.

The sky YAMLs document this in comments; the actual autostop is set at submit
time (it is a launch flag, not a static field), so it is not silently dropped.

## Sweeps: budget preflight + human approval

A sweep multiplies cost by `grid_size = (product of axis lengths) * (#seeds)`,
so `sky/sweep.yaml` is the most budget-sensitive task. Submitting it requires,
in order:

1. a dry-run (`ga launch --dry-run +exp=... sweep=...`) that renders the full
   grid and estimated total cost;
2. a passing budget preflight (the Tier-1 gates in `pyproject.toml`'s
   `[tool.guardian]`: per-job + daily caps, allowed provider, allowed data
   class, clean tree, exact SHA, prior dry-run); and
3. explicit human approval (`--yes`, or `sky jobs launch -y`).

Do not bypass `ga launch` for sweeps — that is where these gates live.

## Tracking (self-hosted, not SaaS)

By default, tracking is a plain local file store (`file:./mlruns`,
`conf/tracking/local.yaml`) and needs no server. For a shared, always-on UI,
`mlflow/docker-compose.yml` runs a **self-hosted** MLflow server with a sqlite
backend and a local artifact directory:

```bash
docker compose -f infra/mlflow/docker-compose.yml up -d
export MLFLOW_TRACKING_URI=http://localhost:5000   # UI at :5000
```

Tracking stays on infrastructure you control — it is intentionally **not** a
managed tracking SaaS. No private/persona data is ever logged (synthetic only),
and the server's entire state lives in one inspectable, backuppable host folder.
