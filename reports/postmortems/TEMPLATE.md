# Postmortem — <experiment> / <sweep or run name>

> Copy this file to `reports/postmortems/<YYYY-MM-DD>-<experiment>-<slug>.md` and fill it
> in after a run or sweep finishes. One postmortem per decision point. Keep it short and
> honest: the point is to record what we *measured* and what we *decided*, not to argue
> that we succeeded. A clean null/negative result is a successful postmortem.

- **Date:**
- **Author / agent:**
- **Experiment:** (e.g. `arithmetic_catapult`)
- **Hypothesis:** (link, e.g. `planning/hypotheses/H001-catapult-arithmetic.md`)
- **Runs / report:** (run_ids, `reports/runs/<name>.md`, MLflow run names)
- **Git SHA(s):** (and whether the tree was clean — dirty-tree runs are not reproducible)

## 1. Hypothesis under test

State the specific claim, the primary metric, and the pre-registered expected signal —
copied/paraphrased from the linked hypothesis file so this postmortem is self-contained.

- Claim:
- Primary metric:
- Pre-registered expected signal:
- Pre-registered kill criterion:

## 2. Setup

What was actually run. Be precise enough to reproduce.

- Model / size (param count):
- Data (split sizes, hard-split definition, synthetic/public/private — should be
  synthetic or public):
- Schedules / conditions compared (control + variants):
- Compute matching (steps / FLOPs held equal across conditions? how?):
- Seeds:
- Config / overrides (Hydra `+exp=...` line):

## 3. Result

The measured numbers. Tables/plots over prose. Reference the figures in
`reports/figures/<experiment>/`.

- Primary metric, control vs. variants (with seed spread):
- Secondary / diagnostic metrics:
- Plot(s):

## 4. Did the signal appear?

The crux. Answer plainly: **yes / no / inconclusive**, and against the *pre-registered*
signal — not a post-hoc one.

- Did the expected signal appear (e.g. did the curves cross / did the active curve beat
  passive / did dynamic beat static at matched compute)?
- Did any stop / kill condition trigger?
- If "yes", is there a mechanistic story (interpretability, ablation isolating the cause)
  or just a number?

## 5. Threats to validity

Why this result might be wrong or might not generalize. List the real ones.

- Scale (toy / single-seed / under-trained?):
- Confounds (compute not truly matched? variant got more effective steps?):
- Metric gaming (memorization explaining a "hard"-split gain? assent explaining a
  preference "win"? sycophancy/echo?):
- Reproducibility (dirty tree? nondeterminism? unseeded ops?):
- Selection effects (best-of-many configs reported as if pre-registered?):

## 6. Decision / next step

What changes because of this. Tie back to the hypothesis status.

- [ ] Confirmed at this scale → escalate (next sweep / larger scale / mechanistic follow-up)
- [ ] Inconclusive → re-run with (more seeds / matched compute / fix confound: ____)
- [ ] Kill criterion met → **downgrade/reject** the hypothesis; record the demotion
- Concrete next action (and who/what owns it):
- Hypothesis status after this postmortem: (open / downgraded / rejected / supported-at-scale-X)

## 7. Cost spent

Honest accounting against the budget (control plane is free; GPUs are the spend).

- GPU hours / provider / instance:
- Estimated USD this run/sweep:
- Cumulative USD on this hypothesis so far:
- Was a budget ceiling hit? (per-job / total / daily)

---

> Reminder: a postmortem records *implemented-tooling output and a decision*. It must not
> claim a scientific finding the data does not support. Distinguish "the tooling ran and
> produced X" from "we have evidence for the hypothesis".
