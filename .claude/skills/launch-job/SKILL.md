---
name: launch-job
description: Use ONLY when the human has explicitly authorized spending real cloud money to launch a guardian-research job or sweep. Walks the gated propose→validate→approve→dry-run→launch flow with hard money-safety checks. If authorization is missing or vague, this skill stops and asks instead of launching.
---

# launch-job — gated cloud launch (Tier 2, spends real money)

This is the one place the project spends money and sends work outside the laptop.
The whole design exists to **preserve the principal's judgment**, so this skill is
deliberately conservative. The code gates (`common/budget.py`, `ga launch`) enforce
the limits; you must also respect them by intent.

## Authorization gate — check BEFORE doing anything

A real launch requires an **explicit, specific** instruction from the human in the
current conversation that names **what to launch** and a **dollar ceiling**. For
example: *"approve and launch `next_arith_sweep` for up to $15."*

The following are **NOT** sufficient authorization — if you only have these, STOP
and ask for an explicit amount:
- "go ahead", "proceed", "do it", "you have authority", "yes" (to something else),
  or any general delegation.

Never set `GUARDIAN_ALLOW_REAL_LAUNCH=1`, never edit `allow_private_persona_data`,
never launch a `private` data class, never exceed the authorized dollar amount,
and never use more than one GPU, unless the human named that specific thing.

## Procedure (only after a clear, dollar-bounded authorization)

1. **Have a passing proposal.** If none exists, run `/propose-next` first.
   Confirm `ga validate-proposal <p>` PASSES.
2. **Dry-run and show the numbers:**
   ```bash
   ga launch --dry-run --proposal <p>
   ```
   Read back: grid size, **est. total cost**, per-job cost, provider, the worker
   YAML. Confirm `est. total ≤ the human's authorized ceiling` AND ≤ the policy
   caps (per-job $5, total/day $25). If it exceeds either, STOP and report — do
   not shrink-and-launch without re-authorization.
3. **Clean, pinned code.** Ensure `git status` is clean and you're on the intended
   commit (cloud workers clone the exact SHA). Commit first if needed.
4. **Record human approval.** Approval is the human's act:
   ```bash
   ga approve <p> --by <human-name>
   ```
   Ask the human to run this, OR run it only if they explicitly tell you to record
   it under their name. Do not invent an approver. (Approval is bound to the
   proposal's content hash + the current SHA; editing the proposal or moving HEAD
   invalidates it.)
5. **Launch (still gated):**
   ```bash
   ga launch --proposal <p> --yes
   ```
   Show the preflight table. Every check must be ✓ (incl. `proposal_approved`).
   Real submission stays disabled unless `[cloud]` is installed and
   `GUARDIAN_ALLOW_REAL_LAUNCH=1` — **do not** set that env var yourself unless the
   human explicitly tells you to; otherwise the gates pass and the exact
   `sky launch` command is printed for the human to run.
6. **Collect results when the job finishes:**
   ```bash
   ga collect --from $GUARDIAN_ARTIFACT_URI --sha <sha>
   ga analyze --experiment <name> --write reports/runs/<name>.md
   ```

## Hard stops (abort and report)

- Authorization is vague or missing → ask for a specific job + dollar ceiling.
- Estimated cost exceeds the authorized amount or a policy cap.
- Any preflight check is ✗ (dirty tree, no SHA, no dry-run, not approved, etc.).
- Anything involving private data, a new provider/credential, deletion, model
  publishing, or >1 GPU — these are Tier 2 and need their own explicit sign-off.

When in doubt, do the cheap local thing (`/run-local`) and ask. Spending the
$2,000 budget wrongly, or leaking data, is far worse than waiting for a clear yes.
