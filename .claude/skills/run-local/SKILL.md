---
name: run-local
description: Use to run a local CPU experiment or a small local sweep in the guardian-research repo and produce a report — e.g. the modular-grokking sweep (H001), arithmetic schedules, or the dynamic-grokking / persona toys. Tier 0 — spends no money, no GPU, no downloads.
---

# run-local — run a CPU experiment/sweep + report

Tier 0: local only, no spend. Keep runs CPU-fast. After running, always
`ga analyze` and summarize **honestly** (no scientific-success claims).

## Pick the experiment

- **H001 catapult/grokking (recommended cheap testbed):**
  modular arithmetic, where the metric is actually movable by schedule.
  ```bash
  for s in baseline_cosine onecycle_high_lr cyclic_lr cyclic_weight_decay; do
    uv run ga train +exp=arithmetic_modular_grok schedule=$s seed=0
  done
  ```
  To sweep the knobs that matter for grokking (and co-tune the cyclic-WD
  multiplier to `base_wd`):
  ```bash
  uv run ga train +exp=arithmetic_modular_grok schedule=cyclic_weight_decay \
    data.train_frac=0.3 train.weight_decay=0.1 seed=0
  ```
- **Base-10 arithmetic (length split is a GPU test):** `+exp=arithmetic_catapult`
  with `model.pos_encoding=rope`. Expect `hard_ood_acc≈0` at CPU scale (see PM001).
- **H002:** `uv run ga grok +exp=dynamic_grokking`
- **H003:** `uv run ga persona run +exp=persona_dynamic_eval`

## Then analyze + record

```bash
uv run ga analyze --experiment <name> --write reports/runs/<name>.md
```

Report: which metric moved, by how much, and whether it is saturated/pinned/noisy.
If this is a decision point for a hypothesis, write a postmortem from
`reports/postmortems/TEMPLATE.md` and update `reports/latest.md`. If results
suggest a next sweep, hand off to `/propose-next` (do not launch).

Multiple seeds are mandatory before calling anything a signal; a single-seed
result is noise. Commit new runs' reports on a branch (the `runs/` cache itself
is gitignored).
