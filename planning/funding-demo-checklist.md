# Funding-demo checklist

The funding story is **not** "we built a more human AI." It is a narrow, defensible
bundle of **three crossovers** — scientific, product, and safety — each demonstrated with
pre-registered metrics, controls, and honest threats-to-validity. Together they would
argue we have found a new scaling axis (*personal adaptive competence*) rather than another
wrapper around an existing model. (Source: `planning/guardian/research-program.md`,
"Budget and milestone logic".)

> Every item below is currently **NOT-YET**. This checklist tracks *what would count as a
> pass*, not what we have shown. A demo is ready only when all three crossovers pass with
> their controls intact and their threats-to-validity addressed in a postmortem.

## Crossover 1 — Scientific (catapult)

**Hypothesis:** `planning/hypotheses/H001-catapult-arithmetic.md`

- **What would count as a pass:** a cyclic / high-LR catapult recipe **eventually beats**
  a `baseline_cosine` control on the **hard arithmetic split** (`final_hard_acc`, ideally
  decomposed into OOD-length and carry-heavy) at **matched parameter count and matched
  total compute** — the curves visibly cross — and the margin **holds across multiple
  seeds**. Bonus / strongest form: winning checkpoints show explicit carry/borrow
  algorithmic structure where the baseline does not.
- **Pass is NOT:** lower final loss; a better *average* or *easy-split* number; a
  single-seed win; a win from tuning many configs and reporting the best.
- **Kill criterion (from H001):** no across-seed hard-split crossover at matched compute →
  downgrade catapult from the headline engine to a side thread.
- **Current status:** **NOT-YET.** Only a CPU smoke snapshot exists
  (`reports/runs/smoke_arithmetic.md`): Δ(best non-baseline − best baseline) on the hard
  split = **+0.000**, single seed, toy scale. This is an unfinished test, not a result.

## Crossover 2 — Product (personalization)

**Hypothesis:** `planning/hypotheses/H003-active-question-personalization.md`

- **What would count as a pass:** an in-weight personalized model (LoRA persona adapter +
  active-question elicitation + replay) **beats both** a prompt-only baseline **and** a
  RAG / long-context baseline on held-out creative-technical tasks for one principal —
  measured by **held-out preference-prediction accuracy** and **edit-distance / revision
  reduction** — and the **active** elicitation loop reaches a target accuracy at a
  **strictly smaller labeled-query budget** than passive finetuning. The margin should be
  obvious in blind pairwise comparison.
- **Pass is NOT:** higher immediate user *assent* (that can be sycophancy/echo); a win that
  needs private persona data (synthetic/public only); memory retrieval alone with no
  preference-aligned behavior.
- **Kill criteria (from H003):** no per-query efficiency edge over passive across seeds →
  reject; gains explained by assent/echo → reject; base capability collapses → broken
  recipe.
- **Current status:** **NOT-YET.** `experiments/persona/` is a stub; no persona corpus,
  no adapter training, no active-question loop, no evaluation has been built or run.

## Crossover 3 — Safety (guardian / robustness)

**Hypotheses:** the security/autonomy thread of
`planning/guardian/guardian-angel.md` ("Too Helpful", "Personality Emulation") and
`planning/guardian/research-program.md` ("Security and autonomy lab"); the matched-FLOPs
"pondering" claim in `planning/hypotheses/H002-dynamic-eval.md` supports the same system.

- **What would count as a pass:** on a small red-team benchmark (out-of-character requests,
  indirect prompt injection, data-exfiltration attempts, sycophancy traps, value-conflict
  cases), the **in-weight personalized** model shows a **lower attack-success / lower
  out-of-character-compliance rate** than (a) a generic assistant and (b) a system-prompt-
  personalized baseline — **without a collapse in utility** — and prefers to **ask a
  clarifying question** rather than obey or bluff on ambiguous/adversarial inputs.
- **Pass is NOT:** "no attacks succeeded" (claim attack-rate *reduction*, never a security
  guarantee); a robustness gain bought by refusing everything (utility must hold);
  anything that promotes untrusted-content-derived weight updates into the persistent
  persona (those must stay quarantined).
- **Kill / honesty stop:** lower attack rate is a *promising property, not proof of
  safety*; interpret conservatively and state it as such in the postmortem.
- **Current status:** **NOT-YET.** No red-team benchmark, no guardian persona, no
  comparison harness exists. The `dynamic_grokking` experiment that would back the
  matched-compute "pondering" demo is also a stub.

## Demo-readiness gate

| Crossover | Owns | Pass metric | Status |
|-----------|------|-------------|--------|
| 1 Scientific | H001 / `arithmetic_catapult` | hard-split crossover at matched compute, multi-seed | **NOT-YET** |
| 2 Product | H003 / `persona` | beats prompt-only + RAG; active > passive per query | **NOT-YET** |
| 3 Safety | H002 + guardian/red-team | lower attack/OOC rate vs. generic, utility held | **NOT-YET** |

**Overall demo status: NOT-YET (0 / 3 crossovers).** Do not present this as a funding demo
until at least the scientific and product crossovers pass with controls and threats-to-
validity documented in `reports/postmortems/`. Claims must stay within what the data
supports — no consciousness, no human-like generalization, no immunity to adversarial
attack, no globally superhuman creativity.
