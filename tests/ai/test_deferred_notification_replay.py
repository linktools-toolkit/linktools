#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable deferred-result notification replay."""

import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from linktools.ai.core import (
    ApprovalDecision,
    ApprovalStatus,
    ExternalCallStatus,
    ResourceKind,
    ResourceRef,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._approval import DefaultApprovalService
from linktools.ai.runtime._external import DefaultExternalService
from linktools.ai.runtime.service_api import (
    ApprovalDecisionRequest,
    ExternalSupplyRequest,
)
from linktools.ai.runtime.state._contracts import ApprovalRecord, ExternalCallRecord
from linktools.ai.storage import ObjectRef
from linktools.ai.workspace import trusted_workspace_principal


class _AllowAuthorization:
    async def authorize(self, principal: object, action: object, resource: object) -> None:
        del principal, action, resource


class _FailOnceGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    async def update_execution(
        self,
        workflow_id: str,
        operation: str,
        payload: dict[str, object],
    ) -> None:
        self.calls.append((workflow_id, operation, payload))
        if len(self.calls) == 1:
            raise RuntimeError("notification transport failed")


class _ExternalCalls:
    def __init__(self, record: ExternalCallRecord) -> None:
        self.record = record

    async def get(self, call_id: str, *, tenant_id: str) -> ExternalCallRecord | None:
        if call_id != self.record.call_id or tenant_id != self.record.tenant_id:
            return None
        return self.record

    async def get_header(self, call_id: str, *, tenant_id: str) -> ResourceRef | None:
        if call_id != self.record.call_id or tenant_id != self.record.tenant_id:
            return None
        return ResourceRef(ResourceKind.EXTERNAL_CALL, call_id, tenant_id)

    async def supply(
        self,
        call_id: str,
        *,
        tenant_id: str,
        expected_status: ExternalCallStatus,
        idempotency_key_digest: str,
        object_ref: ObjectRef,
        payload_digest: str,
        supplied_at: datetime,
    ) -> ExternalCallRecord:
        if (
            call_id != self.record.call_id
            or tenant_id != self.record.tenant_id
            or self.record.status is not expected_status
        ):
            raise AIError(ErrorCode.EXTERNAL_RESULT_CONFLICT)
        self.record = replace(
            self.record,
            status=ExternalCallStatus.SUPPLIED,
            idempotency_key_digest=idempotency_key_digest,
            object_ref=object_ref,
            payload_digest=payload_digest,
            supplied_at=supplied_at,
        )
        return self.record


class _Approvals:
    def __init__(self, record: ApprovalRecord) -> None:
        self.record = record

    async def get(self, approval_id: str, *, tenant_id: str) -> ApprovalRecord | None:
        if approval_id != self.record.approval_id or tenant_id != self.record.tenant_id:
            return None
        return self.record

    async def get_header(self, approval_id: str, *, tenant_id: str) -> ResourceRef | None:
        if approval_id != self.record.approval_id or tenant_id != self.record.tenant_id:
            return None
        return ResourceRef(ResourceKind.APPROVAL, approval_id, tenant_id)

    async def list_pending(self, execution_id: str, *, tenant_id: str) -> tuple[ApprovalRecord, ...]:
        if (
            execution_id == self.record.execution_id
            and tenant_id == self.record.tenant_id
            and self.record.status is ApprovalStatus.PENDING
        ):
            return (self.record,)
        return ()

    async def decide(
        self,
        approval_id: str,
        *,
        tenant_id: str,
        expected_status: ApprovalStatus,
        idempotency_key_digest: str,
        decision: ApprovalDecision,
        principal_id: str,
        decision_digest: str,
        decided_at: datetime,
    ) -> ApprovalRecord:
        if (
            approval_id != self.record.approval_id
            or tenant_id != self.record.tenant_id
            or self.record.status is not expected_status
        ):
            raise AIError(ErrorCode.APPROVAL_CONFLICT)
        self.record = replace(
            self.record,
            status=(
                ApprovalStatus.APPROVED
                if decision is ApprovalDecision.APPROVE
                else ApprovalStatus.DENIED
            ),
            idempotency_key_digest=idempotency_key_digest,
            decision=decision,
            decided_by=principal_id,
            decision_digest=decision_digest,
            decided_at=decided_at,
        )
        return self.record


@pytest.mark.asyncio
async def test_external_supply_replays_notification_after_durable_commit() -> None:
    principal = trusted_workspace_principal("tenant")
    now = datetime.now(timezone.utc)
    calls = _ExternalCalls(
        ExternalCallRecord(
            call_id="external",
            execution_id="execution",
            tenant_id="tenant",
            operation_id="operation",
            status=ExternalCallStatus.PENDING,
            idempotency_key_digest=None,
            object_ref=None,
            payload_digest=None,
            created_at=now,
            supplied_at=None,
        )
    )
    gateway = _FailOnceGateway()
    service = DefaultExternalService(
        SimpleNamespace(external_calls=calls),
        _AllowAuthorization(),
        gateway,
    )
    reference = ObjectRef("store", "objects/result", "a" * 64, 17)
    request = ExternalSupplyRequest(
        principal,
        "external",
        "supply-key",
        reference,
        "b" * 64,
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        await service.supply("execution", request)

    result = await service.supply("execution", request)

    assert result.object_ref == reference
    assert result.payload_digest == "b" * 64
    assert len(gateway.calls) == 2
    workflow_id, operation, payload = gateway.calls[-1]
    assert workflow_id == "execution"
    assert operation == "supply_external_result"
    assert payload["call_id"] == "external"
    assert payload["idempotency_key"] == "supply-key"
    assert payload["payload_digest"] == "b" * 64
    assert json.loads(str(payload["object_ref"])) == {
        "digest": "a" * 64,
        "key": "objects/result",
        "size": 17,
        "store_id": "store",
    }

    conflicting = ExternalSupplyRequest(
        principal,
        "external",
        "supply-key",
        ObjectRef("store", "objects/other", "c" * 64, 3),
        "d" * 64,
    )
    with pytest.raises(AIError) as error:
        await service.supply("execution", conflicting)
    assert error.value.code is ErrorCode.EXTERNAL_RESULT_CONFLICT


@pytest.mark.asyncio
async def test_approval_replays_notification_after_durable_commit() -> None:
    principal = trusted_workspace_principal("tenant")
    now = datetime.now(timezone.utc)
    approvals = _Approvals(
        ApprovalRecord(
            approval_id="approval",
            execution_id="execution",
            tenant_id="tenant",
            operation_id="operation",
            status=ApprovalStatus.PENDING,
            idempotency_key_digest=None,
            decision=None,
            decided_by=None,
            decision_digest=None,
            created_at=now,
            decided_at=None,
        )
    )
    gateway = _FailOnceGateway()
    service = DefaultApprovalService(
        approvals,
        _AllowAuthorization(),
        gateway,
    )
    request = ApprovalDecisionRequest(
        principal,
        "approval",
        "approval-key",
        ApprovalDecision.APPROVE,
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        await service.decide("execution", request)

    result = await service.decide("execution", request)

    assert result.decision is ApprovalDecision.APPROVE
    assert approvals.record.idempotency_key_digest == hashlib.sha256(
        b"approval-key"
    ).hexdigest()
    assert len(gateway.calls) == 2
    workflow_id, operation, payload = gateway.calls[-1]
    assert workflow_id == "execution"
    assert operation == "approve"
    assert payload["approval_id"] == "approval"
    assert payload["idempotency_key"] == "approval-key"
    assert payload["decision"] == ApprovalDecision.APPROVE.value

    conflicting = ApprovalDecisionRequest(
        principal,
        "approval",
        "approval-key",
        ApprovalDecision.DENY,
    )
    with pytest.raises(AIError) as error:
        await service.decide("execution", conflicting)
    assert error.value.code is ErrorCode.APPROVAL_CONFLICT
