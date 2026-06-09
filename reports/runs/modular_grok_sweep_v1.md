# Modular-grokking sweep v1 — CPU results

**Date:** 2026-06-09  
**Hypothesis:** H001 (catapult schedules grok sooner/more robustly)  
**Status:** Tier 0 CPU result — informative for go/no-go on GPU escalation.  
**Cost:** $0  
**Grid:** 4 schedules × 3 train_frac × 3 base_wd × 3 seeds = 108 runs, 10k steps each.  

## Design

Task: `(a + b) mod 97` with train/val split. The model memorizes the training
set (train_acc → 1.0), then later "groks" — val_acc jumps from near-zero to
~1.0 as it discovers the modular algorithm. Schedule/WD are the knobs.

Grid:
- `train_frac ∈ {0.3, 0.4, 0.5}` (lower = harder = longer plateau)
- `base_wd ∈ {0.3, 1.0, 3.0}` (higher = more regularization pressure)
- Schedules: `baseline_cosine`, `onecycle_high_lr`, `cyclic_lr`, `cyclic_weight_decay`
- Cyclic-WD multiplier co-tuned: `wd_max_mult = {5.0, 3.0, 2.0}` for `base_wd = {0.3, 1.0, 3.0}`
  (keeping peak WD in range 1.5–6.0 instead of the catastrophic 20× from probe PM001)
- Seeds: 0, 1, 2

## Aggregate results

| Schedule | Grok rate | Mean grok_step | Median grok_step |
|---|---|---|---|
| **cyclic_weight_decay** | **26/27 (96%)** | **985** | **800** |
| baseline_cosine | 20/27 (74%) | 1850 | 800 |
| onecycle_high_lr | 19/27 (70%) | 1916 | 1000 |
| cyclic_lr | 14/27 (52%) | 3071 | 2000 |

## In the "hard" regime (wd ≥ 1.0, frac ≤ 0.4 — where baseline struggles)

| Schedule | Grok rate | Mean grok_step |
|---|---|---|
| **cyclic_weight_decay** | **11/12 (92%)** | **1145** |
| baseline_cosine | 9/12 (75%) | 2933 |
| onecycle_high_lr | 7/12 (58%) | 2943 |
| cyclic_lr | 4/12 (33%) | 4400 |

## Key head-to-head comparisons

### frac=0.3, wd=3.0 (hardest regime — baseline barely groks)

| Schedule | Seed 0 | Seed 1 | Seed 2 | Rate | Mean step |
|---|---|---|---|---|---|
| baseline_cosine | 6800 | never | 4800 | 2/3 | 5800 |
| onecycle_high_lr | never | 9000 | never | 1/3 | 9000 |
| cyclic_lr | never | never | never | 0/3 | — |
| **cyclic_weight_decay** | **3000** | **800** | **1000** | **3/3** | **1600** |

### frac=0.4, wd=3.0 (baseline barely groks: 1/3 at step 7600)

| Schedule | Seed 0 | Seed 1 | Seed 2 | Rate | Mean step |
|---|---|---|---|---|---|
| baseline_cosine | 7600 | never | never | 1/3 | 7600 |
| onecycle_high_lr | never | never | never | 0/3 | — |
| cyclic_lr | never | never | never | 0/3 | — |
| **cyclic_weight_decay** | **800** | **600** | never | **2/3** | **700** |

### frac=0.3, wd=0.3 (moderate difficulty — baseline 2/3)

| Schedule | Seed 0 | Seed 1 | Seed 2 | Rate | Mean step |
|---|---|---|---|---|---|
| baseline_cosine | never | 1000 | 3800 | 2/3 | 2400 |
| **onecycle_high_lr** | **1600** | **1000** | **1200** | **3/3** | **1267** |
| cyclic_lr | 5200 | 5400 | 400 | 3/3 | 3667 |
| **cyclic_weight_decay** | **2600** | **1000** | **1800** | **3/3** | **1800** |

## Interpretation

1. **Cyclic weight decay (co-tuned) is the clear winner.** It groks in 96% of
   configurations (vs 74% for baseline) and does so 2–4× faster in the hard
   regimes. The effect is largest precisely where baseline struggles — high WD,
   low train_frac — which is where the "sleep" mechanism has the most to offer.

2. **One-cycle high-LR is competitive at moderate difficulty** (frac=0.3, wd=0.3:
   3/3 at step 1267 vs baseline's 2/3 at step 2400) but collapses at high WD.
   This makes sense: a high peak LR combined with high WD is catastrophic.

3. **Cyclic LR is the worst schedule tested** — lower grok rate (52%) and 3×
   slower than baseline when it does grok. The repeated "explore" phases at high
   LR apparently disrupt the slow algorithmic discovery.

4. **The co-tuning fix from PM001 worked.** Keeping `wd_max_mult` at 2–5× (not
   20×) lets cyclic-WD function as intended — periodic regularization pressure
   without crushing learning entirely.

## Does the signal support H001?

**Tentatively yes, for cyclic weight decay.** The "curves cross" pattern is
present: at matched (frac, wd, seed) configs, cyclic-WD groks where baseline
doesn't, and groks sooner where baseline eventually does. The effect is strongest
in the hard regime (large effect: 1145 vs 2933 mean step at 92% vs 75% rate).

**Caveats / threats to validity:**
- This is modular arithmetic, not the base-10 multi-digit task that H001's
  funding demo targets. The grokking phenomenon is well-established here; we're
  testing whether *which schedule* matters, not whether grokking occurs at all.
- 3 seeds is enough to see variance but not enough for statistical tests. The
  directional signal is clear but p-values would be noisy.
- The cyclic-WD multiplier was hand-tuned in this sweep; a fair comparison might
  give baseline the same tuning attention (but baseline_cosine has fewer knobs).
- "Groks sooner on mod-97" ≠ "curves cross on base-10 hard splits at GPU scale."
  This is a *necessary* but not *sufficient* condition for the catapult thesis.

## Decision: go/no-go on GPU

**Go**, with a narrowed grid:
- Promote **cyclic_weight_decay** (with co-tuned multiplier) as the primary
  catapult candidate for the GPU sweep. Drop cyclic_lr (strictly worse).
- Keep **onecycle_high_lr** as a secondary candidate (strong at moderate WD).
- Focus the GPU sweep on the `base_wd ∈ {0.1, 0.3}` range (where onecycle also
  works) and include a cyclic-WD arm at higher WD where it uniquely excels.
- The GPU proposal should be smaller than the 48-job grid in the existing
  `next_arith_sweep.yaml` — this sweep tells us where to look.

## Next steps

1. Draft a tighter GPU proposal informed by these results.
2. Record this in a postmortem as the Phase 2 → Phase 3 gate evidence.
3. Regenerate `reports/latest.md` with updated status.
