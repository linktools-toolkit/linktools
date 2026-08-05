from linktools.ai.execution.domain import ApprovalDecision, RunStatus
from linktools.ai.acp.codec import AcpCodec


def test_permission_decisions_only_use_runtime_terminal_mapping() -> None:
    assert AcpCodec.encode_stop_reason(RunStatus.COMPLETED) == "end_turn"
    assert AcpCodec.encode_stop_reason(RunStatus.CANCELLED) == "cancelled"
    assert ApprovalDecision.ALLOW.value == "allow"
    assert ApprovalDecision.DENY.value == "deny"
