#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized, durable delivery of externally supplied execution results."""

import hashlib
from datetime import datetime, timezone

from ..core import AuthorizationAction, AuthorizationPolicy, ExternalCallStatus
from ..errors import AIError, ErrorCode
from ._persistence import RuntimeStores
from ._services import ExternalSupplyRequest, ExternalSupplyResult, WorkflowGateway


class DefaultExternalService:
    def __init__(self, persistence: RuntimeStores, authorization: AuthorizationPolicy, workflow_gateway: "WorkflowGateway | None" = None) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._workflow_gateway = workflow_gateway

    async def supply(self, execution_id: str, request: ExternalSupplyRequest) -> ExternalSupplyResult:
        call = await self._persistence.recovery.external_calls.get(request.call_id, tenant_id=request.principal.tenant_id)
        if call is None or call.execution_id != execution_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        header = await self._persistence.recovery.external_calls.get_header(request.call_id, tenant_id=request.principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(request.principal, AuthorizationAction.EXTERNAL_SUPPLY, header)
        updated = await self._persistence.recovery.external_calls.supply(
            request.call_id,
            tenant_id=request.principal.tenant_id,
            expected_status=ExternalCallStatus.PENDING,
            idempotency_key_hash=hashlib.sha256(request.idempotency_key.encode("utf-8")).hexdigest(),
            object_ref=request.object_ref,
            payload_digest=request.payload_digest,
            supplied_at=datetime.now(timezone.utc),
        )
        if self._workflow_gateway is not None:
            await self._workflow_gateway.update_execution(
                execution_id,
                "supply_external_result",
                {"call_id": updated.call_id, "idempotency_key": request.idempotency_key, "payload_digest": updated.payload_digest or request.payload_digest, "principal_id": request.principal.principal_id},
            )
        return ExternalSupplyResult(updated.call_id, request.idempotency_key, updated.object_ref or request.object_ref, updated.payload_digest or request.payload_digest)


__all__ = ["DefaultExternalService"]
