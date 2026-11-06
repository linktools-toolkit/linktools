#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized external-result resolution over the deferred frontier."""

import asyncio
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Protocol

from linktools.core import environ
from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    ExternalCallStatus,
    JsonValue,
    Principal,
    ResourceKind,
    ResourceRef,
    ExecutionStatus,
    canonical_json_bytes,
    canonical_sha256,
    normalize_json_value,
    idempotency_key_digest,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, PayloadPolicy, StoredPayload, payload_fits_inline
from ..persistence.objects import ObjectKeyFactory, put_runtime_object, read_runtime_object
from ..model.messages import encode_tool_return_content
from ..execution import (
    ExternalCallFailed,
    ExternalCallRetry,
    ExternalCallSucceeded,
    ExternalCallView,
    ExternalSupplyRequest,
    ExternalSupplyResult,
)
from ..execution import ExecutionRecord, ExecutionRecoveryState
from ..persistence.contracts import (
    deferred_resource_id,
    ExternalCallRecord,
    ExternalCallRepository,
    ExecutionRepository,
    PendingDeferredCall,
)
from ..persistence import PersistenceDomain


_logger = environ.get_logger("ai.runtime.external")


class _DeferredContinuation(Protocol):
    async def reconcile_deferred(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> None: ...


class DefaultExternalService:
    """List and resolve externally supplied values without owning pending context."""

    def __init__(
        self,
        calls: ExternalCallRepository,
        executions: ExecutionRepository,
        authorization: AuthorizationPolicy,
        *,
        objects: ObjectStore,
        object_key_factory: ObjectKeyFactory,
        payload_policy: PayloadPolicy,
        continuation: _DeferredContinuation | None = None,
    ) -> None:
        self._calls = calls
        self._executions = executions
        self._authorization = authorization
        self._objects = objects
        self._object_key_factory = object_key_factory
        self._payload_policy = payload_policy
        self._continuation = continuation

    async def list(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExternalCallView, ...]:
        execution = await self._waiting_execution(
            execution_id, principal=principal,
            action=AuthorizationAction.EXTERNAL_READ,
        )
        if execution is None or execution.recovery is None or execution.recovery.pending_tools is None:
            return ()
        views: list[ExternalCallView] = []
        for pending in execution.recovery.pending_tools.calls:
            call_id = deferred_resource_id(
                "external-call-v1",
                principal.tenant_id,
                execution_id,
                execution.recovery.pending_tools.source_agent_run_id,
                pending.tool_call_id,
            )
            record = await self._calls.get(call_id, tenant_id=principal.tenant_id)
            if record is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            arguments = await _decode_arguments(self._objects, pending.arguments_payload)
            views.append(
                ExternalCallView(
                    call_id,
                    record.status.value,
                    pending.tool_name,
                    arguments,
                    dict(pending.metadata),
                )
            )
        return tuple(views)

    async def supply(
        self,
        execution_id: str,
        request: ExternalSupplyRequest,
    ) -> ExternalSupplyResult:
        execution = await self._waiting_execution(
            execution_id, principal=request.principal,
            action=AuthorizationAction.EXTERNAL_SUPPLY,
        )
        pending = _pending_call(
            execution,
            request.call_id,
            tenant_id=request.principal.tenant_id,
        )
        if pending is None:
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        record = await self._calls.get(
            request.call_id,
            tenant_id=request.principal.tenant_id,
        )
        if record is None or record.execution_id != execution_id:
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        payload, resolution_kind = await self._resolution_payload(request)
        resolution_metadata = dict(request.metadata)
        key_digest = idempotency_key_digest(request.idempotency_key)
        try:
            updated = await self._calls.supply(
                request.call_id,
                tenant_id=request.principal.tenant_id,
                expected_status=ExternalCallStatus.PENDING,
                idempotency_key_digest=key_digest,
                resolution_kind=resolution_kind,
                result_payload=payload,
                resolution_metadata=resolution_metadata,
                supplied_at=datetime.now(timezone.utc),
            )
        except AIError as error:
            if error.code is not ErrorCode.EXTERNAL_RESULT_CONFLICT:
                raise
            updated = await self._calls.get(
                request.call_id,
                tenant_id=request.principal.tenant_id,
            )
            if not _is_exact_replay(
                updated,
                execution_id=execution_id,
                idempotency_key_digest=key_digest,
                resolution_kind=resolution_kind,
                result_payload=payload,
                metadata=resolution_metadata,
            ):
                raise
        if updated is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info(
            "external result resolved: execution=%s call=%s kind=%s",
            execution_id,
            request.call_id,
            resolution_kind,
        )
        await _reconcile(self._continuation, execution_id, request.principal.tenant_id)
        return ExternalSupplyResult(request.call_id, request.idempotency_key, True)

    async def _waiting_execution(
        self,
        execution_id: str,
        *,
        principal: Principal,
        action: AuthorizationAction,
    ) -> ExecutionRecord | None:
        execution = await self._executions.state_store.read(
            lambda _transaction: self._executions.get(execution_id, tenant_id=principal.tenant_id)
        )
        if execution is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            principal, action,
            ResourceRef(ResourceKind.EXECUTION, execution.execution_id, principal.tenant_id),
        )
        if execution.status is not ExecutionStatus.WAITING_DEFERRED:
            return None
        if (execution.recovery_state is not ExecutionRecoveryState.WAITING
                or execution.recovery is None or execution.recovery.pending_tools is None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return execution

    async def _resolution_payload(
        self,
        request: ExternalSupplyRequest,
    ) -> tuple[StoredPayload | None, str]:
        resolution = request.resolution
        if isinstance(resolution, ExternalCallSucceeded):
            value = encode_tool_return_content(resolution.value)
            raw = canonical_json_bytes(value)
            inline = StoredPayload.inline_json(value)
            if payload_fits_inline(inline, self._payload_policy):
                return inline, "succeeded"
            reference = await put_runtime_object(
                self._objects,
                self._object_key_factory,
                PersistenceDomain.CHECKPOINT,
                request.principal.tenant_id,
                raw,
            )
            return StoredPayload.object(reference), "succeeded"
        if isinstance(resolution, ExternalCallRetry):
            return StoredPayload.inline_text(resolution.message), "retry"
        if isinstance(resolution, ExternalCallFailed):
            return StoredPayload.inline_text(resolution.message), "failed"
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _pending_call(
    execution: ExecutionRecord | None,
    call_id: str,
    *,
    tenant_id: str,
) -> PendingDeferredCall | None:
    if execution is None or execution.recovery is None or execution.recovery.pending_tools is None:
        return None
    for pending in execution.recovery.pending_tools.calls:
        candidate = deferred_resource_id(
            "external-call-v1",
            tenant_id,
            execution.execution_id,
            execution.recovery.pending_tools.source_agent_run_id,
            pending.tool_call_id,
        )
        if candidate == call_id:
            return pending
    return None


async def _decode_arguments(objects: ObjectStore, payload: StoredPayload) -> JsonValue:
    if payload.kind == "inline":
        if payload.encoding != "json":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            return normalize_json_value(payload.decode())
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if payload.ref is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    raw = await read_runtime_object(objects, payload.ref)
    try:
        value = json.loads(raw.decode("utf-8"))
        return normalize_json_value(value)
    except (UnicodeError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _is_exact_replay(
    record: ExternalCallRecord | None,
    *,
    execution_id: str,
    idempotency_key_digest: str,
    resolution_kind: str,
    result_payload: StoredPayload | None,
    metadata: Mapping[str, JsonValue],
) -> bool:
    return bool(
        record is not None
        and record.execution_id == execution_id
        and record.status is ExternalCallStatus.SUPPLIED
        and record.idempotency_key_digest == idempotency_key_digest
        and record.resolution_kind == resolution_kind
        and record.result_payload == result_payload
        and dict(record.resolution_metadata) == dict(metadata)
    )


async def _reconcile(
    continuation: _DeferredContinuation | None,
    execution_id: str,
    tenant_id: str,
) -> None:
    if continuation is None:
        return
    try:
        await continuation.reconcile_deferred(execution_id, tenant_id=tenant_id)
    except asyncio.CancelledError:
        raise


__all__ = ["DefaultExternalService"]
