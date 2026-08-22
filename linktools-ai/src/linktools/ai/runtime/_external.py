#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized, durable delivery of externally supplied execution results."""

import hashlib
from datetime import datetime, timezone

from ..core import (
    AuthorizationAction,
    AuthorizationPolicy,
    ExternalCallStatus,
    canonical_json_bytes,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef
from .state._contracts import ExternalCallRecord, RecoveryState
from .service_api import ExternalSupplyRequest, ExternalSupplyResult, WorkflowGateway


class DefaultExternalService:
    def __init__(
        self,
        state: RecoveryState,
        authorization: AuthorizationPolicy,
        workflow_gateway: "WorkflowGateway | None" = None,
    ) -> None:
        self._state = state
        self._authorization = authorization
        self._workflow_gateway = workflow_gateway

    async def supply(
        self,
        execution_id: str,
        request: ExternalSupplyRequest,
    ) -> ExternalSupplyResult:
        if not _is_digest(request.payload_digest):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        call = await self._state.external_calls.get(
            request.call_id,
            tenant_id=request.principal.tenant_id,
        )
        if call is None or call.execution_id != execution_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        header = await self._state.external_calls.get_header(
            request.call_id,
            tenant_id=request.principal.tenant_id,
        )
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(
            request.principal,
            AuthorizationAction.EXTERNAL_SUPPLY,
            header,
        )
        idempotency_digest = hashlib.sha256(
            request.idempotency_key.encode("utf-8")
        ).hexdigest()
        try:
            updated = await self._state.external_calls.supply(
                request.call_id,
                tenant_id=request.principal.tenant_id,
                expected_status=ExternalCallStatus.PENDING,
                idempotency_key_digest=idempotency_digest,
                object_ref=request.object_ref,
                payload_digest=request.payload_digest,
                supplied_at=datetime.now(timezone.utc),
            )
        except AIError as error:
            if error.code is not ErrorCode.EXTERNAL_RESULT_CONFLICT:
                raise
            updated = await self._state.external_calls.get(
                request.call_id,
                tenant_id=request.principal.tenant_id,
            )
            if not _is_exact_replay(
                updated,
                execution_id=execution_id,
                idempotency_key_digest=idempotency_digest,
                object_ref=request.object_ref,
                payload_digest=request.payload_digest,
            ):
                raise
        if updated.object_ref is None or updated.payload_digest is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if self._workflow_gateway is not None:
            await self._workflow_gateway.update_execution(
                execution_id,
                "supply_external_result",
                {
                    "operation_id": updated.operation_id,
                    "call_id": updated.call_id,
                    "idempotency_key": request.idempotency_key,
                    "object_ref": _object_ref_token(updated.object_ref),
                    "payload_digest": updated.payload_digest,
                    "principal_id": request.principal.principal_id,
                },
            )
        return ExternalSupplyResult(
            updated.call_id,
            request.idempotency_key,
            updated.object_ref,
            updated.payload_digest,
        )


def _is_exact_replay(
    record: "ExternalCallRecord | None",
    *,
    execution_id: str,
    idempotency_key_digest: str,
    object_ref: ObjectRef,
    payload_digest: str,
) -> bool:
    return bool(
        record is not None
        and record.execution_id == execution_id
        and record.status is ExternalCallStatus.SUPPLIED
        and record.idempotency_key_digest == idempotency_key_digest
        and record.object_ref == object_ref
        and record.payload_digest == payload_digest
    )


def _object_ref_token(reference: ObjectRef) -> str:
    return canonical_json_bytes(
        {
            "store_id": reference.store_id,
            "key": reference.key,
            "digest": reference.digest,
            "size": reference.size,
        }
    ).decode("utf-8")


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = ["DefaultExternalService"]
