"""Human approval records for proposals.

Approval is the principal's judgment checkpoint in the loop
(propose -> validate -> APPROVE -> launch). An approval is bound to:

* the exact proposal *content* (sha256 of the file) — editing the proposal after
  approval invalidates it; and
* the exact *commit* (git SHA at approval time) — approving code you didn't read
  doesn't count.

The record is a sibling file ``<proposal>.approved.json``. Launch refuses to run
a ``--proposal`` whose approval is missing, stale (proposal edited), or for a
different commit.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from ..common.env_info import current_sha
from .validate_proposal import load_proposal, validate_proposal


def proposal_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def approval_path(proposal_path: str | Path) -> Path:
    p = Path(proposal_path)
    return p.with_suffix(p.suffix + ".approved.json")


def write_approval(proposal_path: str | Path, by: str, note: str = "") -> dict:
    """Validate the proposal, then record approval. Raises if validation fails."""
    report = validate_proposal(load_proposal(proposal_path))
    if not report.passed:
        failed = [c.name for c in report.checks if not c.passed]
        raise ValueError(f"cannot approve a proposal that fails validation: {failed}")
    record = {
        "proposal": str(Path(proposal_path)),
        "proposal_sha256": proposal_sha256(proposal_path),
        "approved_by": by,
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_sha(),
        "note": note,
        "validation_passed": True,
    }
    approval_path(proposal_path).write_text(json.dumps(record, indent=2))
    return record


def read_approval(proposal_path: str | Path) -> dict | None:
    path = approval_path(proposal_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def approval_status(proposal_path: str | Path) -> tuple[bool, str]:
    """Return (valid, reason). Valid = record exists, matches content + commit."""
    rec = read_approval(proposal_path)
    if rec is None:
        return False, "no approval record (run `ga approve`)"
    if rec.get("proposal_sha256") != proposal_sha256(proposal_path):
        return False, "proposal was edited after approval (hash mismatch) — re-approve"
    if rec.get("git_sha") != current_sha():
        return False, f"approved at a different commit ({rec.get('git_sha', '')[:10]}) — re-approve at HEAD"
    return True, f"approved by {rec.get('approved_by')} at {rec.get('approved_at')}"
