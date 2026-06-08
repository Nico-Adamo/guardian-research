# H002 — Dynamic evaluation finds solutions static sampling does not, at equal compute

- **ID:** H002
- **Status:** open (experiment is an honest stub; harness not yet implemented)
- **Source:** `planning/guardian/llm-catapult.md` — Appendix "Dynamic Grokking";
  `planning/guardian/guardian-angel.md` — "Continual Learning", "Brittle Because Fast";
  `planning/guardian/research-program.md` — "Dynamic grokking lab"
- **Owns experiment:** `dynamic_grokking`
  (`src/guardian_research/experiments/dynamic_grokking/`, runner key `dynamic_grokking`)
- **Primary metric:** `solve_rate` at matched FLOPs (dynamic-eval loop vs. static sampling)

> Pre-registered claim and kill criteria. The `dynamic_grokking` module is currently a
> stub (`__init__.py` only); this file says what we *would* measure once it is built. No
> result is claimed. See `reports/latest.md` for current state.

## Claim

For a hard problem an LLM fails to solve by ordinary sampling, **spending compute on
repeated dynamic evaluation** — alternating (a) a few weight updates to a restricted
adapter/last-layer slice on the problem statement + failed traces + any test feedback,
and (b) a fresh rollout ("ponder") — will solve problems that **static sampling cannot
solve at the same total compute (FLOPs)**.

This is the AI analogue of "thinking about a problem for a long time, getting nowhere,
then having an insight": online weight updates accumulate a change that pure forward-pass
search (more samples, self-consistency, text-only self-repair) throws away each turn.

## Metric

- **Primary:** `solve_rate` — fraction of held-out hard problems solved, **as a function
  of total compute**, with compute matched between the dynamic-eval condition and the
  static-sampling condition (this matched-FLOPs accounting is the whole point; an
  unmatched comparison is meaningless).
- **Secondary:** `time_to_first_solve` (serial steps to first correct rollout);
  `solved_only_by_dynamic` — problems solved by the dynamic loop that the static budget
  never solved (the cleanest evidence); per-problem update count at solve.
- **Diagnostic:** post-update `train`/replay loss (did "sleep"/WD steps prevent
  catastrophic forgetting of base capability?).

## Expected signal

At equal FLOPs, the dynamic-eval `solve_rate` curve **rises above** the
static-sampling curve, and there exists a non-empty `solved_only_by_dynamic` set:
problems where the static budget plateaus but the dynamic loop, after N inner updates,
"changes its mind" and produces a correct completion the base model never emitted across
the equivalent number of static samples.

## Ablation / control

- **Control 1 (the key one): static sampling at matched FLOPs** — spend the same compute
  on repeated sampling + self-consistency + text-only self-repair, no weight updates.
- **Control 2: single dynamic step vs. many** — to test the essay's claim that *N* small
  steps ≠ one big step ("more looks let the model change its mind").
- **Control 3: shuffled/irrelevant update text** — update on unrelated tokens to confirm
  any gain is from problem-relevant adaptation, not generic warm-up.
- **Forgetting check:** measure base-task accuracy before/after; an apparent solve that
  destroys general capability is not a win.

## STOP CONDITIONS

- **Per-problem kill:** cap inner dynamic-eval steps per problem (compute ceiling); stop
  on solve or on cap. No unbounded "ponder forever".
- **Forgetting kill:** if dynamic updates collapse base capability (replay loss blows up,
  base-task accuracy craters) without a matching solve-rate gain, stop — the update path
  is broken, not the model.
- **No-edge kill (the kill criterion):** if, at matched FLOPs across the problem set,
  dynamic eval does **not** beat static sampling AND `solved_only_by_dynamic` is empty,
  then **reject H002 at this scale** — "pondering" buys nothing static search doesn't, and
  dynamic grokking is demoted from a headline claim to a curiosity.
- **Observability caveat (pre-registered honesty):** the source essay warns there is no
  reliable internal "is it grokking yet" signal. We therefore commit to judging H002 only
  on the *external* matched-FLOPs solve-rate, and to NOT claiming success from suggestive
  internal metrics alone.
- **Safety stop:** any weight update derived from untrusted/external content stays in a
  quarantined adapter and is never promoted; tools run sandboxed only (cross-ref H003 and
  the security notes in `research-program.md`).
