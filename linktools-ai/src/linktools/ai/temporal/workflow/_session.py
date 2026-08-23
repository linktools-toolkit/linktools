"""Deterministic Session mutation workflow boundary."""

from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

try:
    from temporalio import workflow as _temporal_workflow
    from temporalio.common import RetryPolicy as _TemporalRetryPolicy
except ModuleNotFoundError as error:
    if error.name != "temporalio":
        raise
    _temporal_workflow = None
    _TemporalRetryPolicy = None

from ...core import validate_tenant_id
from ...errors import AIError

SESSION_OPERATIONS = frozenset({"create", "update", "fork", "close"})


@dataclass(frozen=True, slots=True)
class SessionWorkflowInput:
    session_id: str
    tenant_id: str
    expected_revision: int
    operation_id: str
    operation: str

    def __post_init__(self) -> None:
        try:
            validate_tenant_id(self.tenant_id)
        except AIError as error:
            raise ValueError("session workflow tenant is invalid") from error
        if not self.session_id or not self.operation_id or self.expected_revision < 0 or self.operation not in SESSION_OPERATIONS:
            raise ValueError("session workflow input is incomplete")


@dataclass(frozen=True, slots=True)
class SessionWorkflowResult:
    session_id: str
    operation_id: str
    status: str


class SessionActivity(Protocol):
    async def run(self, request: SessionWorkflowInput) -> SessionWorkflowResult: ...


class SessionWorkflow:
    def __init__(self, activity: 'SessionActivity | None' = None) -> None:
        self._activity = activity

    async def run(self, request: SessionWorkflowInput) -> SessionWorkflowResult:
        if self._activity is not None:
            result = await self._activity.run(request)
        elif _temporal_workflow is not None and _TemporalRetryPolicy is not None:
            result = await _temporal_workflow.execute_activity(
                "session_mutation",
                request,
                start_to_close_timeout=timedelta(seconds=60),
                retry_policy=_TemporalRetryPolicy(maximum_attempts=3),
            )
        else:
            raise RuntimeError("Temporal SDK is required for production workflow execution")
        if result.session_id != request.session_id or result.operation_id != request.operation_id:
            raise ValueError("session activity returned mismatched identity")
        return result


if _temporal_workflow is not None:
    SessionWorkflow.run = _temporal_workflow.run(SessionWorkflow.run)
    SessionWorkflow = _temporal_workflow.defn(name="SessionWorkflow")(SessionWorkflow)


__all__ = ["SessionActivity", "SessionWorkflow", "SessionWorkflowInput", "SessionWorkflowResult"]
