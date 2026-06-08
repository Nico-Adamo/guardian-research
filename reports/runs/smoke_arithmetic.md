# Experiment report: arithmetic_catapult

- runs ingested: **8**
- git SHAs present: 038706649c, unknown
- ⚠️ some runs were produced from a **dirty** git tree (not reproducible)

> This report summarizes *measured tooling output*. It makes no claim about whether any hypothesis was confirmed — see `planning/hypotheses/` for the claims and their pre-registered stop conditions.

## Run summary

| run_id                                  | schedule         |   seed | git_sha    | status    |   final_train_loss |   final_easy_acc |   final_hard_acc |   final_train_acc |   final_memorization_gap |   num_params |   final_hard_ood_acc |   final_hard_carry_acc |
|:----------------------------------------|:-----------------|-------:|:-----------|:----------|-------------------:|-----------------:|-----------------:|------------------:|-------------------------:|-------------:|---------------------:|-----------------------:|
| baseline_cosine-20260608T225117-965494  | baseline_cosine  |      0 | unknown    | completed |          2.065     |                0 |           0      |           0       |                  0       |       152400 |                      |                        |
| baseline_cosine-20260608T225142-d7e32d  | baseline_cosine  |      0 | unknown    | completed |          2.065     |                0 |           0      |           0       |                  0       |       152400 |                      |                        |
| baseline_cosine-20260608T230049-9f87b7  | baseline_cosine  |      0 | 038706649c | completed |          2.065     |                0 |           0      |           0       |                  0       |       152400 |                      |                        |
| baseline_cosine-20260608T230118-79c04d  | baseline_cosine  |      0 | 038706649c | completed |          1.056     |                0 |           0      |           0.07812 |                  0.07812 |       152400 |                      |                        |
| baseline_cosine-20260608T230213-5b5936  | baseline_cosine  |      0 | 038706649c | completed |          0.0001118 |                1 |           0      |           1       |                  0       |       152300 |                      |                        |
| baseline_cosine-20260608T230729-8828d3  | baseline_cosine  |      0 | 038706649c | completed |          1.603     |                0 |           0      |           0       |                  0       |       152300 |                    0 |                      0 |
| baseline_cosine-20260608T230857-f827cd  | baseline_cosine  |      0 | 038706649c | completed |          0.0001124 |                1 |           0.3333 |           1       |                  0       |       152300 |                    0 |                      1 |
| onecycle_high_lr-20260608T230929-267afc | onecycle_high_lr |      0 | 038706649c | completed |          0.0003328 |                1 |           0.3333 |           1       |                  0       |       152300 |                    0 |                      1 |

## Plots

### loss

![loss](../figures/arithmetic_catapult/loss.png)

### accuracy

![accuracy](../figures/arithmetic_catapult/accuracy.png)

### memorization_gap

![memorization_gap](../figures/arithmetic_catapult/memorization_gap.png)

### optim

![optim](../figures/arithmetic_catapult/optim.png)

### grad_norm

![grad_norm](../figures/arithmetic_catapult/grad_norm.png)

## Catapult crossover check (hard split)

- best baseline hard-accuracy: **0.333** (baseline_cosine/seed0)
- best non-baseline hard-accuracy: **0.333** (onecycle_high_lr/seed0)
- Δ(best non-baseline − best baseline) = **+0.000** → the baseline is **not yet** beaten on the hard split.

> Interpretation guard: this is a small-scale, single-report snapshot. A real crossover claim (per H001) requires matched compute, multiple seeds, and the pre-registered stop conditions in `planning/hypotheses/H001-catapult-arithmetic.md`.
