#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval decision API and durable default implementation."""

from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ

from ..core import ApprovalStatus, AuthorizationAction, AuthorizationPolicy, Principal, ResourceKind, ResourceRef
from ..errors import AIError, ErrorCode
from ._persistence import RuntimeStores
from ._services import (
    ApprovalDecisionRequest,
    ApprovalDecisionResult,
    ApprovalView,
    WorkflowGateway,
)

_logger = environ.get_logger("ai.runtime.approval")


class ApprovalQueryApi(Protocol):
    async def list(self, execution_id: str, *, principal: Principal) -> 'tuple[ApprovalView, ...]': ...


class ApprovalApi(ApprovalQueryApi, Protocol):
    async def decide(self, execution_id: str, request: ApprovalDecisionRequest) -> ApprovalDecisionResult: ...


class DefaultApprovalService:
    """Persist decisions before any optional workflow notification."""

    def __init__(self, persistence: RuntimeStores, authorization: AuthorizationPolicy, workflow_gateway: "WorkflowGateway | None" = None) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._workflow_gateway = workflow_gateway

    async def list(self, execution_id: str, *, principal: Principal) -> tuple[ApprovalView, ...]:
        await self._authorization.authorize(
            principal,
            AuthorizationAction.APPROVAL_READ,
            ResourceRef(ResourceKind.APPROVAL, execution_id, principal.tenant_id),
        )
        records = await self._persistence.recovery_approval.list_pending(execution_id, tenant_id=principal.tenant_id)
        return tuple(ApprovalView(record.approval_id, record.status) for record in records)

    async def decide(self, execution_id: str, request: ApprovalDecisionRequest) -> ApprovalDecisionResult:
        record = await self._persistence.recovery_approval.get(request.approval_id, tenant_id=request.principal.tenant_id)
        if record is None or record.execution_id != execution_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        header = await self._persistence.recovery_approval.get_header(request.approval_id, tenant_id=request.principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(request.principal, AuthorizationAction.APPROVAL_DECIDE, header)
        decision_digest = _decision_digest(request)
        updated = await self._persistence.recovery_approval.decide(
            request.approval_id,
            tenant_id=request.principal.tenant_id,
            expected_status=ApprovalStatus.PENDING,
            decision_id=request.decision_id,
            decision=request.decision,
            principal_id=request.principal.principal_id,
            decision_digest=decision_digest,
            decided_at=datetime.now(timezone.utc),
        )
        if self._workflow_gateway is not None:
            await self._workflow_gateway.update_execution(
                execution_id,
                "approve",
                {"approval_id": updated.approval_id, "decision_id": updated.decision_id or request.decision_id, "decision": (updated.decision or request.decision).value, "principal_id": request.principal.principal_id, "decision_digest": updated.decision_digest or decision_digest},
            )
        _logger.info("approval decided: execution=%s approval=%s", execution_id, updated.approval_id)
        return ApprovalDecisionResult(updated.approval_id, updated.decision_id or request.decision_id, updated.decision or request.decision)

def _decision_digest(request: ApprovalDecisionRequest) -> str:
    from ..core import canonical_sha256
    return canonical_sha256({"approval_id": request.approval_id, "decision_id": request.decision_id, "decision": request.decision.value, "principal_id": request.principal.principal_id})


__all__ = ["ApprovalApi", "ApprovalQueryApi", "DefaultApprovalService"]
