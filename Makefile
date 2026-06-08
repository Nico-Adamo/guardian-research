# Common commands. `just` is not required (it's optional); these Make targets
# are the portable fallback and mirror the justfile exactly.
.PHONY: setup test smoke lint format clean ui report propose dry-run-sweep help

UV ?= uv

help:
	@echo "Targets: setup test smoke lint format report dry-run-sweep propose ui clean"

setup:                ## create the environment
	$(UV) sync

test:                 ## run the fast (CPU, non-slow) test suite
	$(UV) run pytest -q -m "not gpu and not slow"

smoke:                ## a fast CPU-only toy training run + report (no GPU)
	$(UV) run ga train +exp=arithmetic_catapult model=tiny_transformer schedule=baseline_cosine seed=0 \
		train.max_steps=60 train.eval_every=30 train.eval_n=32 train.log_every=20 data.n_train=800
	$(UV) run ga analyze --experiment arithmetic_catapult --write reports/runs/smoke_arithmetic.md

lint:
	$(UV) run ruff check src tests

format:
	$(UV) run ruff format src tests

report:
	$(UV) run ga analyze --experiment arithmetic_catapult --write reports/runs/arithmetic_catapult.md

dry-run-sweep:        ## render the first cloud sweep without spending money
	$(UV) run ga launch --dry-run --provider skypilot +exp=arithmetic_catapult sweep=arith_lr_wd_seed_v0

propose:              ## draft + validate a next sweep (no launch)
	$(UV) run ga propose --experiment arithmetic_catapult --budget-usd 25 --write reports/proposals/next_arith_sweep.yaml
	$(UV) run ga validate-proposal reports/proposals/next_arith_sweep.yaml

ui:                   ## local MLflow UI (self-hosted, no SaaS)
	$(UV) run mlflow ui --backend-store-uri file:./mlruns

clean:
	rm -rf runs/* artifacts/* mlruns/* reports/figures/* 2>/dev/null || true
