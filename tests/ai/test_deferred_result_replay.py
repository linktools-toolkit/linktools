#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for durable deferred-result replay."""

import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from linktools.ai.core import (
    ApprovalDecision,
    ApprovalStatus,
    canonical_sha256,
    ExternalCallStatus,
    Principal,
    ResourceKind,
    ResourceRef,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._approval import (
    DefaultApprovalService,
    approval_id_for_call,
)
from linktools.ai.runtime._external import (
    DefaultExternalService,
    external_call_id_for_call,
)
from linktools.ai.runtime.service_api import (
    ApprovalDecisionRequest,
    ExternalCallSucceeded,
    ExternalSupplyRequest,
)
from linktools.ai.runtime.state._contracts import (
    ApprovalRecord,
    ExternalCallRecord,
    PendingDeferredCall,
    PendingToolContinuation,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
)
from linktools.ai.storage import InMemoryObjectStore, PayloadPolicy, StoredPayload
from linktools.ai.runtime._object import RuntimeObjectKeyFactory
from linktools.ai.workspace import trusted_workspace_principal


class _AllowAuthorization:
    async def authorize(
        self,
        principal: object,
        action: object,
        resource: object,
    ) -> None:
        del principal, action, resource


class _Executions:
    def __init__(self) -> None:
        self.execution = SimpleNamespace(status=type("Status", (), {"value": "WAITING_DEFERRED"})())

    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None

    async def get(self, execution_id: str, *, tenant_id: str) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return self.execution
        return None


class _Checkpoints:
    def __init__(self, pending: PendingDeferredCall, *, approvals: bool) -> None:
        self.record = RecoveryCheckpoint(
            execution_id="execution",
            tenant_id="tenant",
            step_run_id="step",
            agent_run_sequence=1,
            state=RecoveryCheckpointState.WAITING,
            revision=0,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            pending_tools=PendingToolContinuation(
                "step",
                approvals=(pending,) if approvals else (),
                calls=() if approvals else (pending,),
            ),
        )

    async def get(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> RecoveryCheckpoint | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return self.record
        return None


class _ExternalCalls:
    def __init__(self, record: ExternalCallRecord) -> None:
        self.record = record

    async def get(
        self,
        call_id: str,
        *,
        tenant_id: str,
    ) -> ExternalCallRecord | None:
        if call_id != self.record.call_id or tenant_id != self.record.tenant_id:
            return None
        return self.record

    async def get_header(
        self,
        call_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
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
        resolution_kind: str,
        result_payload: StoredPayload | None,
        resolution_metadata: dict[str, object],
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
            supplied_at=supplied_at,
            resolution_kind=resolution_kind,
            result_payload=result_payload,
            resolution_metadata=resolution_metadata,
        )
        return self.record


class _Approvals:
    def __init__(self, record: ApprovalRecord) -> None:
        self.record = record

    async def get(
        self,
        approval_id: str,
        *,
        tenant_id: str,
    ) -> ApprovalRecord | None:
        if approval_id != self.record.approval_id or tenant_id != self.record.tenant_id:
            return None
        return self.record

    async def get_header(
        self,
        approval_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if approval_id != self.record.approval_id or tenant_id != self.record.tenant_id:
            return None
        return ResourceRef(ResourceKind.APPROVAL, approval_id, tenant_id)

    async def list_pending(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ApprovalRecord, ...]:
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
        decision_message: str | None = None,
        resolution_metadata: dict[str, object] | None = None,
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
            decision_message=decision_message,
            resolution_metadata=(
                {} if resolution_metadata is None else resolution_metadata
            ),
        )
        return self.record


@pytest.mark.asyncio
async def test_external_supply_exact_replay_uses_durable_result() -> None:
    principal = trusted_workspace_principal("tenant")
    now = datetime.now(timezone.utc)
    pending = PendingDeferredCall(
        "tool-call",
        "external_tool",
        StoredPayload.inline_json({"value": 1}),
    )
    call_id = external_call_id_for_call("tenant", "execution", "step", pending.tool_call_id)
    calls = _ExternalCalls(
        ExternalCallRecord(
            call_id=call_id,
            execution_id="execution",
            tenant_id="tenant",
            status=ExternalCallStatus.PENDING,
            idempotency_key_digest=None,
            created_at=now,
            supplied_at=None,
        )
    )
    executions = _Executions()
    checkpoints = _Checkpoints(pending, approvals=False)
    service = DefaultExternalService(
        calls,
        executions,
        checkpoints,
        _AllowAuthorization(),
        objects=InMemoryObjectStore(),
        object_key_factory=object.__new__(RuntimeObjectKeyFactory),
        payload_policy=PayloadPolicy(),
    )
    request = ExternalSupplyRequest(
        principal,
        call_id,
        "supply-key",
        ExternalCallSucceeded({"result": "ok"}),
    )

    first = await service.supply("execution", request)
    second = await service.supply("execution", request)

    assert first == second
    assert second.accepted is True
    assert calls.record.result_payload == StoredPayload.inline_json({"result": "ok"})

    conflicting = ExternalSupplyRequest(
        principal,
        call_id,
        "supply-key",
        ExternalCallSucceeded({"result": "different"}),
    )
    with pytest.raises(AIError) as error:
        await service.supply("execution", conflicting)
    assert error.value.code is ErrorCode.EXTERNAL_RESULT_CONFLICT


@pytest.mark.asyncio
async def test_approval_exact_replay_requires_same_actor() -> None:
    first_principal = Principal("approver-1", "tenant", "service")
    other_principal = Principal("approver-2", "tenant", "service")
    now = datetime.now(timezone.utc)
    pending = PendingDeferredCall(
        "approval-call",
        "approval_tool",
        StoredPayload.inline_json({"value": 1}),
    )
    approval_id = approval_id_for_call(
        "tenant", "execution", "step", pending.tool_call_id
    )
    approvals = _Approvals(
        ApprovalRecord(
            approval_id=approval_id,
            execution_id="execution",
            tenant_id="tenant",
            status=ApprovalStatus.PENDING,
            idempotency_key_digest=None,
            decision=None,
            decided_by=None,
            decision_digest=None,
            created_at=now,
            decided_at=None,
        )
    )
    executions = _Executions()
    checkpoints = _Checkpoints(pending, approvals=True)
    service = DefaultApprovalService(
        approvals,
        executions,
        checkpoints,
        _AllowAuthorization(),
        objects=InMemoryObjectStore(),
    )

    first = await service.decide(
        "execution",
        ApprovalDecisionRequest(
            first_principal,
            approval_id,
            "approval-key",
            ApprovalDecision.APPROVE,
        ),
    )
    persisted_digest = approvals.record.decision_digest
    second = await service.decide(
        "execution",
        ApprovalDecisionRequest(
            first_principal,
            approval_id,
            "approval-key",
            ApprovalDecision.APPROVE,
        ),
    )

    assert first == second
    assert approvals.record.idempotency_key_digest == hashlib.sha256(
        b"approval-key"
    ).hexdigest()
    assert approvals.record.decided_by == first_principal.principal_id
    assert approvals.record.decision_digest == persisted_digest

    with pytest.raises(AIError) as actor_error:
        await service.decide(
            "execution",
            ApprovalDecisionRequest(
                other_principal,
                approval_id,
                "approval-key",
                ApprovalDecision.APPROVE,
            ),
        )
    assert actor_error.value.code is ErrorCode.APPROVAL_CONFLICT
    assert approvals.record.decided_by == first_principal.principal_id
    assert approvals.record.decision_digest == persisted_digest

    conflicting = ApprovalDecisionRequest(
        first_principal,
        approval_id,
        "approval-key",
        ApprovalDecision.DENY,
    )
    with pytest.raises(AIError) as error:
        await service.decide("execution", conflicting)
    assert error.value.code is ErrorCode.APPROVAL_CONFLICT
