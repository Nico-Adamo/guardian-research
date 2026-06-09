# H001 — Cyclic high-LR / high-WD schedules improve HARD arithmetic scaling

- **ID:** H001
- **Status:** open (tooling implemented; not yet tested at a scale that could confirm or refute)
- **Source:** `planning/guardian/llm-catapult.md` — "Prototyping With Arithmetic", "Cyclical Learning Rates"
- **Owns experiments:**
  - `arithmetic_catapult` (base-10 addition; runner `arithmetic`) — the **length-extrapolation** test (`final_hard_ood_acc`). This is a **GPU-scale** test (see probe below).
  - `arithmetic_modular_grok` (modular arithmetic; runner `modular`) — the **cheap CPU grokking** testbed (`final_val_acc`, `grok_step`, `grok_gap`), the canonical memorize→generalize transition from the grokking literature.
- **Primary metric:** `final_hard_ood_acc` (base-10, GPU) and `final_val_acc` / `grok_step` (modular, CPU)

> This file pre-registers a falsifiable claim and its kill criteria. It describes
> *what we will measure and what would count as evidence*, not a result. As of this
> writing nothing here is confirmed; see `reports/latest.md` for what has actually been run.

## Claim

Training a small decoder-only transformer on synthetic arithmetic with a **cyclic /
high-learning-rate schedule** (one-cycle high-LR, cyclic LR, or constant-high-LR +
cyclic weight decay) will, given enough steps at matched compute, **improve accuracy
on a HARD held-out split** (longer-than-trained operands and carry-chain-heavy
examples) relative to a `baseline_cosine` control — even if the catapult recipe looks
*worse* on average / on the easy split for most of training.

The mechanism (per the source essay): high-LR cycles act as regularization that
catapults the model out of the nearest "memorization" basin into a flatter, more
algorithmic basin that implements something closer to true carry/borrow arithmetic,
which is what the hard split is built to expose.

## Metric

- **Primary:** `final_hard_acc` — exact-match accuracy on the hard split (mix of
  out-of-distribution operand lengths and carry-heavy in-distribution lengths).
- **Decomposed:** `final_hard_ood_acc` (longer operands) and `final_hard_carry_acc`
  (carry-chain-heavy at trained lengths) — to see *which* kind of hardness moves.
- **Diagnostic, not target:** `final_easy_acc`, `final_train_acc`,
  `final_memorization_gap` (train_acc − easy_eval_acc), loss spikes, grad-norm.
- **Scaling framing (the real test):** plot `hard_acc` vs compute/steps for each
  recipe. The claim is about the **shape of the curve / the exponent on the hard
  split**, not a single final number. The interesting quantity is *whether the
  catapult curve eventually crosses the baseline curve.*

## Expected signal

**"The curves cross."** Early in training the catapult recipe under-performs the
baseline on easy and average metrics (it is busy escaping basins, with visible loss
spikes at each LR peak). Later, the catapult recipe's hard-split accuracy keeps
climbing while the baseline's plateaus, and at some step the catapult `final_hard_acc`
**exceeds** the baseline `final_hard_acc` and stays above it. A confirmatory signal
would additionally show the gain is concentrated in `hard_ood_acc` / `hard_carry_acc`
(true extrapolation), not just easy-split noise.

## Ablation / control

- **Control:** `baseline_cosine` (standard cosine-decay LR, low/no cycling) at
  **matched parameter count and matched total optimizer steps**.
- **Isolate the factor:** sweep `schedule` × `train.lr` × `train.weight_decay` × `seed`
  so we can attribute any hard-split gain to peak-LR, to WD cycling, or to their
  interaction — rather than to "tried more configs". (See
  `conf/sweep/arith_lr_wd_seed_v0.yaml` and `reports/proposals/next_arith_sweep.yaml`.)
- **Multiple seeds** are mandatory: a single-seed crossover is noise, not signal.
- **Negative control of the hard split itself:** the hard split must be genuinely OOD
  (the data generator guarantees hard operand lengths are strictly longer than train
  lengths), so memorization cannot explain a gain.

## Testbeds & initial probes (honest, CPU-scale)

Two probes were run to find a regime where the metric can actually *move* (see
`reports/postmortems/PM001-position-and-grokking-probes.md`):

1. **Base-10 length extrapolation is pinned at CPU scale.** Training on 1–3 digit
   addition and testing 4–5 digits gives `hard_ood_acc = 0.000` for **all** of
   `learned`, `none`, and `rope` positional schemes (rope learns in-distribution
   best at 0.98 but still does not extrapolate). Length generalization on addition
   needs wider training ranges / scale — so `arithmetic_catapult`'s OOD metric is a
   **GPU-scale** experiment (`+exp=arithmetic_catapult_gpu model=small_transformer`,
   which now defaults to `rope` so extrapolation is at least *possible*). The
   in-distribution / carry-heavy splits saturate to ~1.0 for every schedule, so they
   do **not** discriminate cheaply.
2. **Modular arithmetic is the movable CPU testbed.** On `(a+b) mod 97`, the metric
   is strongly schedule-sensitive: `baseline_cosine` (wd=1.0) groks to `val_acc=1.0`
   by ~step 600, while `cyclic_weight_decay` at its default 20× multiplier
   (= peak wd≈20 here) over-regularizes and never learns (`val_acc=0.028`). That the
   final metric ranges 0.03↔1.00 by schedule is the point: H001 is **testable** here.
   The immediate lesson — WD-schedule multipliers are relative to `base_wd` and must
   be co-tuned — is itself a knob for the sweep, not a result.

The canonical *delayed* grok (long memorization plateau, then a late val jump) needs
`train_frac` / `wd` / model-size tuning; finding the regime where a high-LR/cyclic
schedule groks **sooner or more robustly** than baseline is the H001 research the
propose→sweep loop now drives.

## STOP CONDITIONS

Per-shard (cheap, automatic):
- Kill a shard if `train_loss` is NaN/Inf, or if `train_acc < 0.05` after 50% of steps
  (the run never left the floor — not a catapult, just a dead run).

Sweep-level (decision points):
- **No-crossover kill (the kill criterion):** if, after running all pre-registered
  seeds at matched compute, **no** non-baseline schedule beats the best baseline
  `final_hard_acc` by a margin that holds across seeds, then **downgrade H001**: stop
  treating "catapult pretraining" as the main engine and demote it to a side thread.
  (The current smoke snapshot in `reports/runs/smoke_arithmetic.md` shows
  Δ = +0.000 — i.e. not yet beaten — at toy scale; this is *not* a refutation, only an
  un-finished test, because compute and seed coverage are far below what the claim needs.)
- **Scale-honesty stop:** do not escalate to GPU base-10 length extrapolation until a
  schedule effect is observed on the **cheap modular CPU testbed** (where the metric is
  movable). If no schedule reliably groks sooner/more robustly than baseline there,
  more compute on base-10 is unlikely to be the missing ingredient.
- **Budget stop:** respect the per-job and total-cost ceilings in the active proposal
  (`reports/proposals/next_arith_sweep.yaml`); halt the sweep at the total-cost ceiling
  regardless of result.

## Follow-up if (and only if) the signal appears

Mechanistic confirmation: inspect the winning checkpoints for explicit carry/borrow
structure (per `experiments/arithmetic/analyze.py` + the essay's pointer to Zhong et al
2023 / interpretability). A crossover *without* an algorithmic story is weaker evidence
and should be flagged as such in the postmortem (`reports/postmortems/`).
