# Local setup

This guide gets you from a fresh checkout to a passing test suite and a working
CPU smoke run. Everything here is **CPU-only and offline** — the default install
downloads no models and needs no GPU or cloud account.

For the *why* behind the local-control-plane / disposable-worker design, see
[`../planning/decisions/ADR-0001-experiment-factory.md`](../planning/decisions/ADR-0001-experiment-factory.md)
and [`../planning/decisions/ADR-0002-disposable-gpu-workers.md`](../planning/decisions/ADR-0002-disposable-gpu-workers.md).

---

## 1. Prerequisites

- **Python 3.11** (the project supports `>=3.10,<3.13`; the venv is built on 3.11).
- **[`uv`](https://docs.astral.sh/uv/)** for environment + dependency management.
  Install it once:

  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # then restart your shell or:  export PATH="$HOME/.local/bin:$PATH"
  ```

- **`just`** (optional). Every `just` recipe is mirrored exactly in the
  `Makefile`, so you can use `make` instead. Install `just` only if you prefer
  it (`brew install just`, `cargo install just`, etc.).
- **git**. Used for nothing exotic locally, but cloud workers identify code by
  exact commit SHA, so a clean tree matters before any launch.

You do **not** need: a GPU, a CUDA toolkit, a Hugging Face account, or any cloud
credentials to run the core, the tests, or the smoke run.

---

## 2. Create the environment

```bash
uv sync
```

This resolves and installs the **lean core** from `pyproject.toml`:
PyTorch (CPU build is fine), NumPy, Hydra/OmegaConf, Pydantic v2, pandas,
matplotlib, MLflow, Rich, and the dev tools (`pytest`, `ruff`, `pre-commit`).
The core is deliberately small so `uv sync` is fast and the first deliverable
runs with no downloads.

Equivalent recipe: `just setup` or `make setup`.

The `ga` console script is installed into the environment; invoke it with
`uv run ga ...` (or activate the venv with `source .venv/bin/activate` and call
`ga` directly).

---

## 3. Verify the checkout

```bash
just test     # or: make test    →  uv run pytest -q -m "not gpu and not slow"
just smoke    # or: make smoke   →  tiny CPU training run + markdown report
```

- **`test`** runs the fast suite (CPU, non-slow). It exercises the arithmetic
  data generator, config composition, the budget guard, the result schema, the
  propose→validate loop, and a CPU smoke-train. There are GPU/slow markers
  defined in `pyproject.toml`; the default selection skips them.
- **`smoke`** trains a tiny transformer on synthetic arithmetic for a few
  hundred steps and writes `reports/runs/smoke_arithmetic.md`. It is the
  canonical "is my environment sane?" check and finishes in well under a minute
  on a laptop CPU.

If both pass, you have a working control plane. Try the full milestone sequence
in the [README](../README.md#the-milestone-command-sequence).

---

## 4. Local tracking UI (optional)

```bash
just ui    # or: make ui   →  uv run mlflow ui --backend-store-uri file:./mlruns
```

Opens the MLflow UI against the local **file store** at `file:./mlruns`. Nothing
is uploaded anywhere; the store is a gitignored directory in the repo. Every
`ga train` already writes a self-contained run directory under `runs/` *and*
logs to this store. MLflow is imported lazily and degrades to a no-op if it is
ever absent, so the core never breaks without it.

---

## 5. Optional extras

The core install covers all default (CPU, offline) code paths. Heavier or
network-dependent tooling is gated behind extras so it never slows down the
common case. Install only what you need:

### `cloud` — submit real cloud jobs

```bash
uv sync --extra cloud
```

Installs SkyPilot. **Not required for `ga launch --dry-run`**, which only renders
YAML and commands and spends nothing. You only need this to actually submit a
job — and even then a real launch *additionally* requires
`GUARDIAN_ALLOW_REAL_LAUNCH=1`, a clean git tree at a known SHA, a prior
dry-run, a passing budget preflight, and an explicit `--yes`. See
[`security.md`](security.md) for the full gate.

### `llm` — real language-model personalization

```bash
uv sync --extra llm
```

Installs `transformers`, `peft`, `datasets`, and `accelerate` for the
(non-toy) Guardian personalization / dynamic-evaluation work. **This extra is
the only thing that can pull model weights from the network**, so it is opt-in
and never touched by tests or the default smoke path. Even with it installed,
only **synthetic** personas are permitted by policy — see
[Data handling](#7-data-handling).

### `dvc` — data/artifact versioning

```bash
uv sync --extra dvc
```

Installs DVC. The repo's `data/` layout (`raw/`, `processed/`, `splits/`) is
**DVC-compatible without DVC installed** — DVC is purely optional. Use it if you
want to version generated datasets, hard splits, and report artifacts with git
metadata while keeping the large files out of git.

You can combine extras: `uv sync --extra cloud --extra dvc`.

---

## 6. Configuring secrets (`.env`)

Copy the template and fill in only what you need:

```bash
cp .env.example .env     # .env is gitignored — NEVER commit it
```

Key fields (all optional for local CPU work):

- `GUARDIAN_REPO_URL` — the **read-only** git URL a disposable worker clones.
- `GUARDIAN_ALLOW_REAL_LAUNCH` — must be `1` (plus the `[cloud]` extra and a
  passing preflight) before `ga launch --yes` can submit anything.
- `GUARDIAN_MAX_JOB_COST_USD` / `GUARDIAN_MAX_DAILY_COST_USD` — env overrides
  that can only **tighten** the caps in `pyproject.toml`, never loosen them.
- Provider credentials (`RUNPOD_API_KEY`, `LAMBDA_API_KEY`, `MODAL_TOKEN_*`) —
  used by SkyPilot; keep secret.
- `MLFLOW_TRACKING_URI` — defaults to `file:./mlruns`; change only if you run a
  remote tracking endpoint.
- `GUARDIAN_ALLOW_PRIVATE_PERSONA_DATA` — defaults to `false`; leave it off.

Secrets are read **only** from the environment or a local `.env`. They are never
hard-coded, never committed, and never shipped to a worker (workers receive code
via git@SHA only).

---

## 7. Data handling

- The repo uses **synthetic and public data only**. The arithmetic track
  generates data on the fly; personas are synthetic.
- The budget policy (`[tool.guardian]` in `pyproject.toml`) restricts
  `allowed_data_classes` to `["public", "synthetic"]` and sets
  `allow_private_persona_data = false`. The `private` data class is gated and
  off by default.
- **No private persona data anywhere** — not in tests, toys, or on workers.

---

## 8. Troubleshooting

**`ga: command not found`**
The script lives in the venv. Use `uv run ga ...`, or activate the venv:
`source .venv/bin/activate`.

**`No runner set` when running `ga train`**
You forgot the experiment. A bare config cannot train; pass `+exp=...`, e.g.
`uv run ga train +exp=arithmetic_catapult model=tiny_transformer schedule=baseline_cosine seed=0`.

**Hydra "Could not override" / "key not found"**
You are overriding a key that is not in the composed config, or using `key=val`
where a new key needs `+key=val`. Group overrides (`model=`, `schedule=`,
`launcher=`, `tracking=`, `sweep=`) select files under the matching `conf/<group>/`
directory; experiments are *added* with `+exp=<name>`. List available options by
browsing `conf/`.

**`uv sync` is slow or fails to resolve PyTorch**
The core pins `torch>=2.2,<3`; a CPU wheel is sufficient. If your platform needs
a specific index, configure `uv` per the
[uv docs](https://docs.astral.sh/uv/) — the project does not require a CUDA
build for any default path.

**MLflow UI shows nothing**
You have not run anything yet, or you are pointing at the wrong store. Run
`just smoke` first, then `just ui` (which uses `file:./mlruns`).

**A test marked `gpu` or `slow` got selected**
The default `make test` / `just test` deselects them with
`-m "not gpu and not slow"`. If you invoke pytest directly, add that marker
filter.

**`ga launch` did something unexpected**
By design it defaults to `--dry-run` and spends nothing. If you intended a real
launch and it refused, read the printed preflight report — one of the gates
(budget, clean tree, exact SHA, dry-run-first, `GUARDIAN_ALLOW_REAL_LAUNCH`,
`--yes`, or the `[cloud]` extra) is not satisfied. This is the safety system
working as intended; see [`security.md`](security.md).

**Resetting local state**
`just clean` (or `make clean`) wipes `runs/`, `artifacts/`, `mlruns/`, and
`reports/figures/`. It does not touch your source or configs.
