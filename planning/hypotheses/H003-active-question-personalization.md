# H003 — Active-question personalization predicts held-out preferences more efficiently than passive finetuning

- **ID:** H003
- **Status:** open (experiment is an honest stub; harness not yet implemented)
- **Source:** `planning/guardian/guardian-angel.md` — "Active Learning", "Preference
  Learning", "Personality Emulation", "Initial Steps / GBT";
  `planning/guardian/research-program.md` — "Guardian personalization lab"
- **Owns experiment:** `persona`
  (`src/guardian_research/experiments/persona/`, runner key `persona`,
  entry `train_persona.run`)
- **Primary metric:** held-out preference-prediction accuracy **per labeled query**
  (sample-efficiency), active vs. passive

> Pre-registered claim and kill criteria. The `persona` module is currently a stub
> (`__init__.py` only); this file specifies the evaluation before the code exists. No
> result is claimed, and **only synthetic / public persona data is ever used** — never
> real private persona data. See `reports/latest.md`.

## Claim

A persona model that **actively asks the principal the highest-uncertainty questions**
(active learning / elicitation, using simple seed/checkpoint-ensemble disagreement as the
uncertainty estimate, then finetuning a LoRA-style persona adapter on the answers) will
reach a given **held-out preference-prediction accuracy with fewer labeled
queries** than the same model finetuned passively on an equal *amount* of corpus data
chosen at random.

Per the source essay: random/passive offline data suffers a "curse of exploration"
(daily life is predictable and quickly uninformative), whereas learner-chosen contrastive
questions can drive error down far faster — DAgger-style, exponential rather than
square-root in the favorable case.

## Metric

- **Primary:** held-out **preference-prediction accuracy as a function of the number of
  labeled queries consumed** (the x-axis is *query budget*, not wall-clock). Sample
  efficiency = accuracy reached per query. Compare the active curve to the passive curve.
- **Secondary:** edit-distance / revision reduction on held-out "what would the principal
  write/choose?" tasks; calibration of the ensemble's uncertainty (are the questions it
  picks actually the informative ones?).
- **Diagnostic:** base-capability retention with replay (personalization must not destroy
  general competence).

## Expected signal

At every query budget the **active** curve sits **above** the **passive** curve on
held-out preference prediction, and reaches the target accuracy at a **strictly smaller
query budget** — i.e. the same held-out performance for materially fewer principal
questions. A confirmatory signal: the questions the active loop chose are the ones whose
answers most changed the model's held-out predictions (informative, not random).

## Ablation / control

- **Control 1 (the key one): passive finetuning** on randomly-sampled corpus items of
  equal data quantity — same model, same adapter, same compute; only the *selection* of
  what to label differs.
- **Control 2: in-context / RAG-only baseline** (no weight update) to confirm the gain
  needs in-weight personalization, not just retrieval.
- **Control 3: random-question active loop** (ask randomly instead of by uncertainty) to
  isolate that the *active selection*, not merely "asking questions", is what helps.
- **Sycophancy / echo guard (control + threat):** evaluate on *prediction of future
  held-out choices*, never on immediate user assent. A model that just agrees with the
  principal can win an assent metric while failing the held-out-prediction metric — reward
  better disagreement, not mirroring.

## STOP CONDITIONS

- **Query-budget cap:** fixed maximum number of labeled queries; the comparison is run to
  that cap and stopped. No unbounded elicitation.
- **Forgetting kill:** if persona finetuning collapses base capability beyond a small
  pre-set tolerance (with replay enabled), stop — the adapter recipe is broken.
- **Sycophancy kill:** if the active model's gains are explained by assent/echoing (high
  immediate-agreement but no held-out preference-prediction improvement over passive),
  **reject the result** — that is an overfit echo, not a guardian.
- **No-efficiency-edge kill (the kill criterion):** if, at matched query budget, the
  active loop does **not** beat passive finetuning on held-out preference prediction across
  seeds, then **reject H003 at this scale** — active elicitation provides no sample-
  efficiency advantage here, and we fall back to passive corpus finetuning.
- **Privacy stop (hard rule):** synthetic/public personas only; no real private persona
  data enters the repo or any run, ever. A run that would require private data is not run.
