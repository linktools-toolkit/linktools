#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval creation public-contract regression coverage."""

import pytest

from linktools.ai.core import (
    ApprovalDecision,
    ApprovalStatus,
    OperationStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    TenantAuthorizationPolicy,
    canonical_sha256,
    idempotency_key_digest,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import (
    ApprovalCreateRequest,
    ApprovalDecisionRequest,
    DefaultApprovalService,
    RuntimeState,
)


class _Executions:
    async def get_header(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ResourceRef | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None


@pytest.mark.asyncio
async def test_approval_create_is_atomic_idempotent_and_replayable() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="approval-test", tenant_id="tenant")
    try:
        principal = Principal("caller", "tenant", "service")
        service = DefaultApprovalService(
            state.recovery.approvals,
            _Executions(),
            TenantAuthorizationPolicy("tenant"),
        )
        request = ApprovalCreateRequest(
            principal,
            "approval",
            "business-operation",
            "create-key",
        )

        first = await service.create("execution", request)
        second = await service.create("execution", request)

        assert first == second
        assert first.status is ApprovalStatus.PENDING
        record = await state.recovery.approvals.get(
            "approval",
            tenant_id="tenant",
        )
        assert record is not None
        assert record.operation_id == "business-operation"
        assert record.idempotency_key_digest is None

        create_operation_id = canonical_sha256(
            {
                "action": "approval.create.operation",
                "tenant_id": "tenant",
                "execution_id": "execution",
                "idempotency_key_digest": idempotency_key_digest("create-key"),
            }
        )
        operation = await state.recovery.operations.get(
            create_operation_id,
            tenant_id="tenant",
        )
        assert operation is not None
        assert operation.status is OperationStatus.SUCCEEDED
        assert operation.resource_kind is ResourceKind.APPROVAL
        assert operation.resource_id == "approval"
        assert operation.result_ref == "approval"
        assert "create-key" not in repr(operation)

        decided = await service.decide(
            "execution",
            ApprovalDecisionRequest(
                principal,
                "approval",
                "decision-key",
                ApprovalDecision.APPROVE,
            ),
        )
        assert decided.decision is ApprovalDecision.APPROVE

        replay_after_decision = await service.create("execution", request)
        assert replay_after_decision.status is ApprovalStatus.APPROVED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_approval_create_distinguishes_idempotency_and_resource_conflicts() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="approval-conflict-test", tenant_id="tenant")
    try:
        principal = Principal("caller", "tenant", "service")
        service = DefaultApprovalService(
            state.recovery.approvals,
            _Executions(),
            TenantAuthorizationPolicy("tenant"),
        )
        await service.create(
            "execution",
            ApprovalCreateRequest(
                principal,
                "approval",
                "business-operation",
                "create-key",
            ),
        )

        with pytest.raises(AIError) as error:
            await service.create(
                "execution",
                ApprovalCreateRequest(
                    principal,
                    "approval",
                    "different-operation",
                    "create-key",
                ),
            )
        assert error.value.code is ErrorCode.IDEMPOTENCY_CONFLICT

        with pytest.raises(AIError) as error:
            await service.create(
                "execution",
                ApprovalCreateRequest(
                    principal,
                    "approval",
                    "business-operation",
                    "different-key",
                ),
            )
        assert error.value.code is ErrorCode.APPROVAL_CONFLICT

        with pytest.raises(AIError) as error:
            await service.create(
                "execution",
                ApprovalCreateRequest(
                    principal,
                    "different-approval",
                    "business-operation",
                    "create-key",
                ),
            )
        assert error.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_approval_create_hides_cross_tenant_execution_existence() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="approval-auth-test", tenant_id="tenant")
    try:
        service = DefaultApprovalService(
            state.recovery.approvals,
            _Executions(),
            TenantAuthorizationPolicy("tenant"),
        )
        with pytest.raises(AIError) as error:
            await service.create(
                "execution",
                ApprovalCreateRequest(
                    Principal("caller", "other-tenant", "service"),
                    "approval",
                    "business-operation",
                    "create-key",
                ),
            )
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()
