#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized, durable delivery of externally supplied execution results."""

from datetime import datetime, timezone

from ..core import ErrorCode, AIError
from ..core import AuthorizationAction, AuthorizationPolicy
from ..core import ExternalCallStatus
from ._persistence import RuntimePersistence
from ._services import ExternalResultRequest, ExternalResultResult, WorkflowGateway


class DefaultExternalService:
    def __init__(self, persistence: RuntimePersistence, authorization: AuthorizationPolicy, workflow_gateway: "WorkflowGateway | None" = None) -> None:
        self._persistence = persistence
        self._authorization = authorization
        self._workflow_gateway = workflow_gateway

    async def supply(self, execution_id: str, request: ExternalResultRequest) -> ExternalResultResult:
        header = await self._persistence.executions.get_header(execution_id, tenant_id=request.principal.tenant_id)
        if header is None:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        await self._authorization.authorize(request.principal, AuthorizationAction.EXTERNAL_SUPPLY, header)
        call = await self._persistence.externals.get(request.call_id, tenant_id=request.principal.tenant_id)
        if call is None or call.execution_id != execution_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        updated = await self._persistence.externals.supply(
            request.call_id,
            tenant_id=request.principal.tenant_id,
            expected_status=ExternalCallStatus.PENDING,
            result_id=request.result_id,
            payload_ref=request.payload_ref,
            payload_digest=request.payload_digest,
            supplied_at=datetime.now(timezone.utc),
        )
        if self._workflow_gateway is not None:
            await self._workflow_gateway.update_execution(
                execution_id,
                "supply_external_result",
                {"call_id": updated.call_id, "result_id": updated.result_id or request.result_id, "payload_ref": updated.payload_ref or request.payload_ref, "payload_digest": updated.payload_digest or request.payload_digest, "principal_id": request.principal.principal_id},
            )
        return ExternalResultResult(updated.call_id, updated.result_id or request.result_id, updated.payload_ref or request.payload_ref, updated.payload_digest or request.payload_digest)


__all__ = ["DefaultExternalService"]
