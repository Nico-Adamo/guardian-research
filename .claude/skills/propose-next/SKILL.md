---
name: propose-next
description: Use to draft and validate the next sweep proposal from existing results in guardian-research. Tier 0 — produces an inert YAML proposal and a PASS/FAIL validation; never launches anything.
---

# propose-next — draft + validate a next sweep (no launch)

A proposal is **data, not an action.** This skill drafts one and gates it; it
stops short of approval and launch (those are the human's, via `/launch-job`).

## Steps

1. Make sure there are prior runs to learn from (else run `/run-local` first).
2. Draft:
   ```bash
   uv run ga propose --experiment <name> --budget-usd <N> --write reports/proposals/<name>.yaml
   ```
   `--budget-usd` is the **total** sweep budget the agent may plan within (must
   fit the daily cap). The drafter puts `baseline_cosine` in the schedule axis so
   the comparison is matched-compute.
3. Validate:
   ```bash
   uv run ga validate-proposal reports/proposals/<name>.yaml
   ```
   This checks budget (per-job ≤ $5, total ≤ $25/day), data class (synthetic/public),
   provider allowlist, scientific framing (hypothesis/metric/ablation/stop
   conditions), git SHA, and that the config actually composes in Hydra.
4. If it FAILS: explain which checks failed and fix the proposal (edit axes,
   shrink the grid, correct the budget), then re-validate. Do not approve or
   launch a failing proposal.
5. If it PASSES: present the proposal to the human — hypothesis, sweep grid, job
   count, per-job and total cost — and **STOP**. Tell them the next steps are:
   ```bash
   ga approve reports/proposals/<name>.yaml --by <you>
   ga launch --dry-run --proposal reports/proposals/<name>.yaml   # then /launch-job
   ```

Never edit `pyproject.toml [tool.guardian]` caps to make a proposal pass. Tighten
proposals to fit the policy, not the other way around.
