"""Approval decision API and durable default implementation."""

import hashlib
from datetime import datetime, timezone

from linktools.core import environ

from ..core import (
    ApprovalDecision,
    ApprovalStatus,
    AuthorizationAction,
    AuthorizationPolicy,
    Principal,
)
from ..errors import AIError, ErrorCode
from .service_api import (
    ApprovalDecisionRequest,
    ApprovalDecisionResult,
    ApprovalView,
    WorkflowGateway,
)
from .state._contracts import ApprovalRecord, ApprovalRepository, ExecutionRepository

_logger = environ.get_logger("ai.runtime.approval")


class DefaultApprovalService:
    """Persist decisions before any optional workflow notification."""

    def __init__(
        self,
        approvals: ApprovalRepository,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        workflow_gateway: "WorkflowGateway | None" = None,
    ) -> None:
        self._approvals = approvals
        self._executions = executions
        self._authorization = authorization
        self._workflow_gateway = workflow_gateway

    async def list(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ApprovalView, ...]:
        header = await self._executions.get_header(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            principal,
            AuthorizationAction.APPROVAL_READ,
            header,
        )
        records = await self._approvals.list_pending(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        return tuple(ApprovalView(record.approval_id, record.status) for record in records)

    async def decide(
        self,
        execution_id: str,
        request: ApprovalDecisionRequest,
    ) -> ApprovalDecisionResult:
        record = await self._approvals.get(
            request.approval_id,
            tenant_id=request.principal.tenant_id,
        )
        if record is None or record.execution_id != execution_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        header = await self._approvals.get_header(
            request.approval_id,
            tenant_id=request.principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.APPROVAL_DECIDE,
            header,
        )
        decision_digest = _decision_digest(request)
        idempotency_digest = hashlib.sha256(
            request.idempotency_key.encode("utf-8")
        ).hexdigest()
        try:
            updated = await self._approvals.decide(
                request.approval_id,
                tenant_id=request.principal.tenant_id,
                expected_status=ApprovalStatus.PENDING,
                idempotency_key_digest=idempotency_digest,
                decision=request.decision,
                principal_id=request.principal.principal_id,
                decision_digest=decision_digest,
                decided_at=datetime.now(timezone.utc),
            )
        except AIError as error:
            if error.code is not ErrorCode.APPROVAL_CONFLICT:
                raise
            updated = await self._approvals.get(
                request.approval_id,
                tenant_id=request.principal.tenant_id,
            )
            if not _is_exact_replay(
                updated,
                execution_id=execution_id,
                idempotency_key_digest=idempotency_digest,
                decision=request.decision,
            ):
                raise
        if (
            updated.decision is None
            or updated.decided_by is None
            or updated.decision_digest is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._workflow_gateway is not None:
            await self._workflow_gateway.update_execution(
                execution_id,
                "approve",
                {
                    "operation_id": updated.operation_id,
                    "approval_id": updated.approval_id,
                    "idempotency_key": request.idempotency_key,
                    "decision": updated.decision.value,
                    "principal_id": updated.decided_by,
                    "decision_digest": updated.decision_digest,
                },
            )
        _logger.info(
            "approval decided: execution=%s approval=%s",
            execution_id,
            updated.approval_id,
        )
        return ApprovalDecisionResult(
            updated.approval_id,
            request.idempotency_key,
            updated.decision,
        )


def _is_exact_replay(
    record: "ApprovalRecord | None",
    *,
    execution_id: str,
    idempotency_key_digest: str,
    decision: ApprovalDecision,
) -> bool:
    expected_status = (
        ApprovalStatus.APPROVED
        if decision is ApprovalDecision.APPROVE
        else ApprovalStatus.DENIED
    )
    return bool(
        record is not None
        and record.execution_id == execution_id
        and record.status is expected_status
        and record.idempotency_key_digest == idempotency_key_digest
        and record.decision is decision
    )


def _decision_digest(request: ApprovalDecisionRequest) -> str:
    from ..core import canonical_sha256

    return canonical_sha256(
        {
            "approval_id": request.approval_id,
            "idempotency_key": request.idempotency_key,
            "decision": request.decision.value,
            "principal_id": request.principal.principal_id,
        }
    )


__all__ = ["DefaultApprovalService"]