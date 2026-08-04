from linktools.ai.execution.domain import ApprovalDecision, RunStatus
from linktools.ai.acp.execution import AcpExecutionAdapter


def test_permission_decisions_only_use_runtime_terminal_mapping() -> None:
    assert AcpExecutionAdapter.stop_reason(RunStatus.COMPLETED) == "end_turn"
    assert AcpExecutionAdapter.stop_reason(RunStatus.CANCELLED) == "cancelled"
    assert ApprovalDecision.ALLOW.value == "allow"
    assert ApprovalDecision.DENY.value == "deny"
