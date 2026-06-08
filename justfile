# Common commands (requires `just`; a Makefile mirror exists for environments
# without it). Run `just` to list recipes.
uv := "uv"

default:
    @just --list

# create the environment
setup:
    {{uv}} sync

# run the fast (CPU, non-slow) test suite
test:
    {{uv}} run pytest -q -m "not gpu and not slow"

# a fast CPU-only toy training run + report (no GPU)
smoke:
    {{uv}} run ga train +exp=arithmetic_catapult model=tiny_transformer schedule=baseline_cosine seed=0 \
        train.max_steps=60 train.eval_every=30 train.eval_n=32 train.log_every=20 data.n_train=800
    {{uv}} run ga analyze --experiment arithmetic_catapult --write reports/runs/smoke_arithmetic.md

lint:
    {{uv}} run ruff check src tests

format:
    {{uv}} run ruff format src tests

# render an experiment report
report:
    {{uv}} run ga analyze --experiment arithmetic_catapult --write reports/runs/arithmetic_catapult.md

# render the first cloud sweep without spending money
dry-run-sweep:
    {{uv}} run ga launch --dry-run --provider skypilot +exp=arithmetic_catapult sweep=arith_lr_wd_seed_v0

# draft + validate a next sweep (no launch)
propose:
    {{uv}} run ga propose --experiment arithmetic_catapult --budget-usd 25 --write reports/proposals/next_arith_sweep.yaml
    {{uv}} run ga validate-proposal reports/proposals/next_arith_sweep.yaml

# local MLflow UI (self-hosted, no SaaS)
ui:
    {{uv}} run mlflow ui --backend-store-uri file:./mlruns

clean:
    rm -rf runs/* artifacts/* mlruns/* reports/figures/* 2>/dev/null || true
