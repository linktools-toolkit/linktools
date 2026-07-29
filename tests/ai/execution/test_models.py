from datetime import datetime, timezone

from linktools.ai.execution.domain import ApprovalDecision, RunApproval


def test_run_approval_retains_binding_and_audit() -> None:
    now = datetime.now(timezone.utc)
    approval = RunApproval(
        approval_id="approval-1",
        tool_call_id="call-1",
        tool_name="write_file",
        binding_fingerprint="sha256",
        decision=ApprovalDecision.ALLOW,
        decided_by="user:alice",
        decided_at=now,
    )
    assert approval.binding_fingerprint == "sha256"
    assert approval.decided_at is now
