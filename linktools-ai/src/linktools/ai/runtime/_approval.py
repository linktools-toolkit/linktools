#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval service and durable default implementation."""

import asyncio
from datetime import datetime, timezone

from linktools.core import environ

from ..core import (
    ApprovalDecision,
    ApprovalStatus,
    AuthorizationAction,
    AuthorizationPolicy,
    OperationKind,
    OperationLedgerInput,
    OperationStatus,
    Principal,
    ResourceKind,
    canonical_sha256,
    idempotency_key_digest,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from .service_api import (
    ApprovalCreateRequest,
    ApprovalDecisionRequest,
    ApprovalContextReader,
    ApprovalContinuation,
    ApprovalDecisionResult,
    ApprovalView,
)
from .state import ApprovalRepository, ExecutionRepository
from .state._contracts import ApprovalRecord

_logger = environ.get_logger("ai.runtime.approval")


class DefaultApprovalService:
    """Own Approval creation, observation, and decision semantics."""

    def __init__(
        self,
        approvals: ApprovalRepository,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        *,
        context_reader: ApprovalContextReader | None = None,
        continuation: ApprovalContinuation | None = None,
    ) -> None:
        self._approvals = approvals
        self._executions = executions
        self._authorization = authorization
        self._context_reader = context_reader
        self._continuation = continuation

    async def create(
        self,
        execution_id: str,
        request: ApprovalCreateRequest,
    ) -> ApprovalView:
        tenant_id = request.principal.tenant_id
        header = await self._executions.get_header(
            execution_id,
            tenant_id=tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.APPROVAL_CREATE,
            header,
        )

        key_digest = idempotency_key_digest(request.idempotency_key)
        request_digest = canonical_sha256(
            {
                "action": "approval.create",
                "principal": principal_identity_payload(request.principal),
                "execution_id": execution_id,
                "approval_id": request.approval_id,
                "operation_id": request.operation_id,
            }
        )
        admission_operation_id = canonical_sha256(
            {
                "action": "approval.create.operation",
                "tenant_id": tenant_id,
                "execution_id": execution_id,
                "idempotency_key_digest": key_digest,
            }
        )
        result_digest = canonical_sha256(
            {
                "approval_id": request.approval_id,
                "execution_id": execution_id,
                "operation_id": request.operation_id,
            }
        )
        now = datetime.now(timezone.utc)
        record = ApprovalRecord(
            approval_id=request.approval_id,
            execution_id=execution_id,
            tenant_id=tenant_id,
            operation_id=request.operation_id,
            status=ApprovalStatus.PENDING,
            idempotency_key_digest=None,
            decision=None,
            decided_by=None,
            decision_digest=None,
            created_at=now,
            decided_at=None,
        )
        operation = OperationLedgerInput(
            operation_id=admission_operation_id,
            tenant_id=tenant_id,
            resource_kind=ResourceKind.APPROVAL,
            resource_id=request.approval_id,
            execution_id=execution_id,
            operation_kind=OperationKind.APPROVAL,
            status=OperationStatus.SUCCEEDED,
            request_digest=request_digest,
            result_ref=request.approval_id,
            result_digest=result_digest,
            error_code=None,
            compactable=True,
            created_at=now,
            updated_at=now,
        )
        created, replayed = await self._approvals.create_with_operation(
            record,
            operation=operation,
        )
        _logger.info(
            "approval create admitted: execution=%s approval=%s replayed=%s",
            execution_id,
            created.approval_id,
            replayed,
        )
        return ApprovalView(created.approval_id, created.status)

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
        if self._context_reader is None:
            return tuple(ApprovalView(record.approval_id, record.status) for record in records)
        approval_ids = tuple(record.approval_id for record in records)
        contexts = await self._context_reader.tool_approvals(
            approval_ids,
            execution_id=execution_id,
            tenant_id=principal.tenant_id,
        )
        unexpected = set(contexts) - set(approval_ids)
        if unexpected:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        views: list[ApprovalView] = []
        for record in records:
            context = contexts.get(record.approval_id)
            if context is None:
                views.append(ApprovalView(record.approval_id, record.status))
                continue
            if canonical_sha256(context.arguments) != context.args_digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            views.append(
                ApprovalView(
                    record.approval_id,
                    record.status,
                    kind="tool",
                    tool_name=context.tool_name,
                    arguments=context.arguments,
                )
            )
        return tuple(views)

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
        idempotency_digest = idempotency_key_digest(request.idempotency_key)
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
                principal_id=request.principal.principal_id,
                decision_digest=decision_digest,
                tenant_id=request.principal.tenant_id,
            ):
                raise
        if (
            updated.decision is None
            or updated.decided_by is None
            or updated.decision_digest is None
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "approval decided: execution=%s approval=%s",
            execution_id,
            updated.approval_id,
        )
        if self._continuation is not None:
            try:
                await self._continuation.reconcile_approval(
                    execution_id,
                    tenant_id=request.principal.tenant_id,
                )
            except asyncio.CancelledError:
                raise
            except AIError:
                raise
            except Exception as error:
                _logger.warning(
                    "approval continuation reconciliation failed: execution=%s approval=%s",
                    execution_id,
                    request.approval_id,
                    exc_info=environ.debug,
                )
                raise AIError(
                    ErrorCode.STORAGE_RECOVERY_REQUIRED,
                    retryable=True,
                    safe_details={"phase": "approval_continuation"},
                ) from error
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
    principal_id: str,
    decision_digest: str,
    tenant_id: str,
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
        and record.decided_by == principal_id
        and record.decision_digest == decision_digest
        and record.decided_at is not None
        and record.tenant_id == tenant_id
    )


def _decision_digest(request: ApprovalDecisionRequest) -> str:
    return canonical_sha256(
        {
            "approval_id": request.approval_id,
            "idempotency_key": request.idempotency_key,
            "decision": request.decision.value,
            "principal": principal_identity_payload(request.principal),
        }
    )


__all__ = ["DefaultApprovalService"]
