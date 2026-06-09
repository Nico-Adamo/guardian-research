"""Approval gate + collect dispatch tests (the closed loop's human checkpoint)."""

import pytest
import yaml

from guardian_research.agents.approval import approval_status, write_approval
from guardian_research.agents.propose import propose_sweep
from guardian_research.commands.collect import _pull_cmd


def _write(p, path):
    path.write_text(yaml.safe_dump(p.model_dump(), sort_keys=False))
    return path


def test_approve_binds_to_content_and_commit(tmp_path):
    proposal = propose_sweep("arithmetic_catapult", 25.0)
    path = _write(proposal, tmp_path / "prop.yaml")

    write_approval(path, by="tester", note="looks good")
    ok, reason = approval_status(path)
    assert ok, reason

    # Editing the proposal after approval must invalidate it (hash mismatch).
    path.write_text(path.read_text() + "\n# sneaky edit\n")
    ok2, reason2 = approval_status(path)
    assert not ok2
    assert "edit" in reason2.lower()


def test_cannot_approve_invalid_proposal(tmp_path):
    proposal = propose_sweep("arithmetic_catapult", 25.0)
    proposal.data_class = "private"  # fails validation -> must not be approvable
    path = _write(proposal, tmp_path / "bad.yaml")
    with pytest.raises(ValueError):
        write_approval(path, by="tester")


def test_unapproved_proposal_is_not_valid(tmp_path):
    proposal = propose_sweep("arithmetic_catapult", 25.0)
    path = _write(proposal, tmp_path / "unapproved.yaml")
    ok, reason = approval_status(path)
    assert not ok and "no approval" in reason.lower()


def test_collect_pull_cmd_schemes():
    cmd, tool = _pull_cmd("s3://bucket/guardian", "abc123", "/local/runs/")
    assert tool == "aws" and cmd[:3] == ["aws", "s3", "sync"]
    cmd, tool = _pull_cmd("gs://bucket/guardian", "abc123", "/local/runs/")
    assert tool == "gsutil"
    cmd, tool = _pull_cmd("user@host:/data/guardian", "abc123", "/local/runs/")
    assert tool == "rsync"
