# Research ledger — index

This is the table of contents for the project's **research ledger**: the
human-readable trace of what was hypothesized, what was run, what was measured,
and what was learned. Reports describe *what was measured* and *what tooling
exists*; they never claim a scientific result. Null and negative results are
recorded honestly.

Reports are generated from the authoritative run artifacts
(`runs/<experiment>/<run_id>/results.json`) via `ga analyze`. See
`docs/tracking.md` for how ingestion and report generation work.

## Run reports — [`reports/runs/`](runs/)

Per-experiment markdown reports (comparison tables + plots) rendered from
collected runs.

```bash
uv run ga analyze --experiment <name> --write reports/runs/<name>.md
# Makefile shortcuts:
make report     # arithmetic_catapult → reports/runs/arithmetic_catapult.md
make smoke      # CPU toy run + reports/runs/smoke_arithmetic.md
```

Existing:

- [`runs/smoke_arithmetic.md`](runs/smoke_arithmetic.md) — CPU-only smoke run of
  the arithmetic catapult vertical.
- Plots live under [`reports/figures/`](figures/) (e.g.
  `figures/arithmetic_catapult/{loss,accuracy,memorization_gap,optim,grad_norm}.png`),
  referenced by the run reports.

## Proposals — [`reports/proposals/`](proposals/)

Drafted next sweeps with hypothesis, metric, stop conditions, budget, and exact
reproducibility (`git_sha`, `requires_exact_sha`). Generated and validated
without launching anything:

```bash
uv run ga propose --experiment <name> --budget-usd 25 --write reports/proposals/<name>.yaml
uv run ga validate-proposal reports/proposals/<name>.yaml
# Makefile shortcut: make propose
```

Existing:

- [`proposals/next_arith_sweep.yaml`](proposals/next_arith_sweep.yaml) — proposed
  LR × WD × schedule × seed sweep for `arithmetic_catapult` (H001).

## Postmortems — [`reports/postmortems/`](postmortems/)

Written after a sweep/experiment to record what happened and what was learned —
including honest null/negative outcomes and any decision to downgrade a
hypothesis. (Empty until the first sweep completes.)

## Latest snapshot — `reports/latest.md`

A pointer to the most recent headline report/snapshot. Not yet generated; it will
be produced by `ga analyze` once a full run set exists. Until then, browse
[`reports/runs/`](runs/) directly.

## Hypotheses — [`../planning/hypotheses/`](../planning/hypotheses/)

The falsifiable claims this repo is built to test (e.g. **H001**: high-LR / cyclic
schedules improve HARD arithmetic accuracy vs. a `baseline_cosine` control at
matched compute — *"the curves cross"* on the hard split). Proposals reference
these IDs, and reports/postmortems are where they are confirmed or downgraded.
See also [`../planning/decisions/`](../planning/decisions/) for design decisions
and [`../planning/guardian/`](../planning/guardian/) for the source ideas
(`llm-catapult.md`, `guardian-angel.md`, `research-program.md`).

## How the ledger fits together

```
planning/hypotheses/   →  ga propose   →  reports/proposals/*.yaml
        ↑                                        │
        │                                  ga launch (exact SHA, disposable worker)
        │                                        │
        │                                 runs/<exp>/<id>/results.json   ← source of truth
        │                                        │
        │                                  ga analyze
        │                                        ▼
   (confirm / downgrade)  ←  reports/postmortems/  ←  reports/runs/*.md + reports/figures/
```
