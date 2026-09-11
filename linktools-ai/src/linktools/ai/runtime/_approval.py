#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized approval resolution over the recovery deferred frontier."""

import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ
from ..core import (
    ApprovalDecision,
    ApprovalStatus,
    AuthorizationAction,
    AuthorizationPolicy,
    Principal,
    canonical_sha256,
    idempotency_key_digest,
    normalize_json_value,
    principal_identity_payload,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, StoredPayload
from ._object import read_runtime_object
from .service_api import (
    ApprovalDecisionRequest,
    ApprovalDecisionResult,
    ApprovalView,
)
from .state._contracts import (
    ApprovalRecord,
    ApprovalRepository,
    ExecutionRepository,
    PendingDeferredCall,
    RecoveryCheckpoint,
    RecoveryCheckpointRepository,
    RecoveryCheckpointState,
)


_logger = environ.get_logger("ai.runtime.approval")


class _DeferredContinuation(Protocol):
    async def reconcile_deferred(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> None: ...


class DefaultApprovalService:
    """Resolve human decisions without owning pending tool context."""

    def __init__(
        self,
        approvals: ApprovalRepository,
        executions: ExecutionRepository,
        checkpoints: RecoveryCheckpointRepository,
        authorization: AuthorizationPolicy,
        *,
        objects: ObjectStore,
        continuation: _DeferredContinuation | None = None,
    ) -> None:
        self._approvals = approvals
        self._executions = executions
        self._checkpoints = checkpoints
        self._authorization = authorization
        self._objects = objects
        self._continuation = continuation

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
        checkpoint = await self._waiting_checkpoint(
            execution_id,
            tenant_id=principal.tenant_id,
        )
        if checkpoint is None or checkpoint.pending_tools is None:
            return ()
        views: list[ApprovalView] = []
        for pending in checkpoint.pending_tools.approvals:
            approval_id = approval_id_for_call(
                principal.tenant_id,
                execution_id,
                checkpoint.pending_tools.source_step_run_id,
                pending.tool_call_id,
            )
            record = await self._approvals.get(
                approval_id,
                tenant_id=principal.tenant_id,
            )
            if record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            arguments = await _decode_arguments(self._objects, pending.arguments_payload)
            views.append(
                ApprovalView(
                    approval_id,
                    record.status,
                    tool_name=pending.tool_name,
                    arguments=arguments,
                    metadata=dict(pending.metadata),
                )
            )
        return tuple(views)

    async def decide(
        self,
        execution_id: str,
        request: ApprovalDecisionRequest,
    ) -> ApprovalDecisionResult:
        header = await self._executions.get_header(
            execution_id,
            tenant_id=request.principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.APPROVAL_DECIDE,
            header,
        )
        checkpoint = await self._waiting_checkpoint(
            execution_id,
            tenant_id=request.principal.tenant_id,
        )
        pending = _pending_approval(checkpoint, request.approval_id)
        if pending is None:
            raise AIError(ErrorCode.APPROVAL_CONFLICT)
        record = await self._approvals.get(
            request.approval_id,
            tenant_id=request.principal.tenant_id,
        )
        if record is None or record.execution_id != execution_id:
            raise AIError(ErrorCode.APPROVAL_CONFLICT)
        decision_digest = _decision_digest(request)
        key_digest = idempotency_key_digest(request.idempotency_key)
        try:
            updated = await self._approvals.decide(
                request.approval_id,
                tenant_id=request.principal.tenant_id,
                expected_status=ApprovalStatus.PENDING,
                idempotency_key_digest=key_digest,
                decision=request.decision,
                principal_id=request.principal.principal_id,
                decision_digest=decision_digest,
                decided_at=datetime.now(timezone.utc),
                decision_message=request.message,
                resolution_metadata=request.metadata,
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
                idempotency_key_digest=key_digest,
                decision=request.decision,
                principal_id=request.principal.principal_id,
                decision_digest=decision_digest,
                message=request.message,
                metadata=request.metadata,
            ):
                raise
        if updated is None or updated.decision is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "approval resolved: execution=%s approval=%s decision=%s",
            execution_id,
            request.approval_id,
            updated.decision.value,
        )
        if self._continuation is not None:
            await self._reconcile(execution_id, request.principal.tenant_id)
        return ApprovalDecisionResult(
            updated.approval_id,
            request.idempotency_key,
            updated.decision,
        )

    async def _waiting_checkpoint(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> RecoveryCheckpoint | None:
        execution = await self._executions.get(execution_id, tenant_id=tenant_id)
        if execution is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        checkpoint = await self._checkpoints.get(execution_id, tenant_id=tenant_id)
        if execution.status.value != "WAITING_DEFERRED":
            return None
        if (
            checkpoint is None
            or checkpoint.state is not RecoveryCheckpointState.WAITING
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return checkpoint

    async def _reconcile(self, execution_id: str, tenant_id: str) -> None:
        try:
            await self._continuation.reconcile_deferred(
                execution_id,
                tenant_id=tenant_id,
            )
        except asyncio.CancelledError:
            raise
        except AIError:
            raise
        except Exception as error:
            _logger.warning(
                "deferred approval reconciliation failed: execution=%s",
                execution_id,
                exc_info=environ.debug,
            )
            raise AIError(
                ErrorCode.STORAGE_RECOVERY_REQUIRED,
                retryable=True,
                safe_details={"phase": "approval_reconcile"},
            ) from error


def approval_id_for_call(
    tenant_id: str,
    execution_id: str,
    source_step_run_id: str,
    tool_call_id: str,
) -> str:
    """Return the deterministic id for one approval-owned deferred call."""
    return canonical_sha256(
        {
            "contract": "approval-v1",
            "tenant_id": tenant_id,
            "execution_id": execution_id,
            "source_step_run_id": source_step_run_id,
            "tool_call_id": tool_call_id,
        }
    )


def _pending_approval(
    checkpoint: RecoveryCheckpoint | None,
    approval_id: str,
) -> PendingDeferredCall | None:
    if checkpoint is None or checkpoint.pending_tools is None:
        return None
    for pending in checkpoint.pending_tools.approvals:
        candidate = approval_id_for_call(
            checkpoint.tenant_id,
            checkpoint.execution_id,
            checkpoint.pending_tools.source_step_run_id,
            pending.tool_call_id,
        )
        if candidate == approval_id:
            return pending
    return None


async def _decode_arguments(objects: ObjectStore, payload: StoredPayload) -> object:
    if payload.kind == "inline":
        if payload.encoding != "json":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            return normalize_json_value(payload.decode())
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    try:
        raw = await read_runtime_object(objects, payload.ref)
        return normalize_json_value(json.loads(raw.decode("utf-8")))
    except (UnicodeError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _decision_digest(request: ApprovalDecisionRequest) -> str:
    return canonical_sha256(
        {
            "approval_id": request.approval_id,
            "decision": request.decision.value,
            "message": request.message,
            "metadata": request.metadata,
            "decided_by": principal_identity_payload(request.principal),
        }
    )


def _is_exact_replay(
    record: ApprovalRecord | None,
    *,
    execution_id: str,
    idempotency_key_digest: str,
    decision: ApprovalDecision,
    principal_id: str,
    decision_digest: str,
    message: str | None,
    metadata: Mapping[str, object],
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
        and record.decision_message == message
        and dict(record.resolution_metadata) == dict(metadata)
    )


__all__ = ["DefaultApprovalService"]
