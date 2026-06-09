---
name: next-step
description: Use at the start of a guardian-research session, or whenever the user asks "what's our next step", "where are we", "what should I do", or "status". Orients on the current state (ledger, git, runs, hypotheses, open proposals) and recommends the single highest-value next action — without spending money.
---

# next-step — orient and recommend

Goal: give the principal an accurate, honest read of where the project stands and
**one** recommended next action, plus a couple of alternatives. This is Tier 0:
read-only / local only. Do not launch anything.

## 1. Gather state (run these)

```bash
uv run ga status
git log --oneline -8 ; git status --porcelain
```

Then read (do not skip): `reports/latest.md`, the files in `planning/hypotheses/`,
anything in `reports/proposals/` (and whether a matching `*.approved.json` exists),
the newest entries in `reports/postmortems/`, and `planning/funding-demo-checklist.md`.

## 2. Synthesize (report to the user)

- **Where we are:** what tooling is implemented vs. what has actually been *run*
  (cite `latest.md`). Be explicit that toy/plumbing output is not evidence.
- **Open loop items:** any proposal awaiting validation/approval/launch, and its
  exact gate status (validated? approved? dry-run done?).
- **Per-hypothesis status:** H001/H002/H003 — is the metric movable, what's the
  next experiment, and what stop/kill condition applies.
- **Blockers:** separate *plumbing* blockers from *science/scale* blockers.

## 3. Recommend ONE next action

Pick the highest-leverage, lowest-cost move and say why. Bias toward cheap CPU
work that de-risks before any spend. Current default frontier (update as the repo
evolves): **run the modular-grokking sweep** (`/run-local`) to test whether a
catapult schedule groks sooner/more robustly than baseline; only escalate to the
GPU length experiment after a CPU signal. Offer 2–3 concrete options and let the
human choose. If the best next step would spend money, do **not** do it — describe
it and ask for explicit authorization (see `/launch-job`).

Always distinguish *implemented tooling* from *empirical findings*. Never claim a
hypothesis is confirmed.
