# PM001 — Probes: positional schemes & where the H001 metric can move

- **Date:** 2026-06-08
- **Hypothesis:** [H001](../../planning/hypotheses/H001-catapult-arithmetic.md)
- **Status:** tooling/probe result — **not** a confirmation or refutation of H001.
- **Cost:** $0 (CPU only).

## Why this probe

Before spending any cloud budget, we needed to know whether H001's metric can even
*move* at small scale, or whether it is structurally pinned (in which case a cloud
sweep would be wasted money). Two regimes were probed.

## Setup

- Model: `tiny_transformer` (d=64, 3 layers) / for modular a small classifier head.
- Probe A (length extrapolation): train 1–3 digit base-10 addition, test OOD 4–5
  digits, `hard_ood_frac=1.0`, 2000 steps, lr 3e-3, seed 0, schemes `learned|none|rope`.
- Probe B (modular grokking): `(a+b) mod 97`, `train_frac=0.4`, 6000 steps, wd base 1.0,
  schedules `baseline_cosine` vs `cyclic_weight_decay`, seed 0.

## Results

**Probe A — base-10 length extrapolation (final hard-OOD accuracy):**

| pos_encoding | easy_acc | hard_ood_acc |
|---|---|---|
| learned | 0.883 | **0.000** |
| none | 0.188 | **0.000** |
| rope | 0.984 | **0.000** |

→ No positional scheme extrapolates to longer operands at this scale. `rope` learns
the in-distribution task best but still cannot generalize length. The carry-heavy /
in-distribution splits separately saturate to ~1.0 for every schedule.

**Probe B — modular grokking (final accuracy):**

| schedule | final_train | final_val | grok_step |
|---|---|---|---|
| baseline_cosine | 1.000 | **1.000** | 600 |
| cyclic_weight_decay | 0.036 | **0.028** | never |

→ The metric is **strongly schedule-sensitive** (0.03 ↔ 1.00). The `cyclic_weight_decay`
schedule's default `wd_max_mult=20` — calibrated for the base-10 regime's `base_wd≈0.05`
— becomes a catastrophic peak `wd≈20` at the modular `base_wd=1.0`, crushing learning.

## Did the signal appear?

- H001's length-OOD signal: **not observable at CPU scale** (pinned at 0). Reclassified
  as a **GPU-scale** experiment; `small_transformer` now defaults to `rope` so OOD is at
  least possible there.
- A schedule *effect* on the modular testbed: **yes, large** — but the current effect is
  "cyclic-WD as configured destroys learning", i.e. a config-interaction finding, not a
  catapult win. The canonical delayed-grok regime (and whether a cyclic schedule groks
  *sooner* than baseline) still needs tuning.

## Threats to validity

- Single seed; tiny model; one (p, train_frac) point. Not evidence for/against H001.
- "baseline groks at step 600" shows little memorization delay at these settings — the
  dramatic grok plateau needs lower `train_frac` / tuned `wd`.

## Decision / next step

1. Keep length-OOD as the GPU experiment; gate it behind the modular CPU result.
2. Sweep the modular testbed: `train_frac ∈ {0.3,0.4,0.5}` × `schedule` × `base_wd`
   (co-tuning `cyclic_weight_decay` multipliers to `base_wd`) × seeds, to find a regime
   with a real memorization plateau and to test "catapult groks sooner".
3. Only then consider escalating base-10 to GPU.
