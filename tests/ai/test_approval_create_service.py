#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1 approval service surface and authorization regressions."""

import pytest

from linktools.ai.core import (
    ApprovalDecision,
    ExecutionStatus,
    Principal,
    ResourceKind,
    ResourceRef,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ApprovalDecisionRequest
from linktools.ai.runtime._approval import DefaultApprovalService


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

    async def get(self, execution_id: str, *, tenant_id: str) -> object:
        del execution_id, tenant_id
        return type("Execution", (), {"status": ExecutionStatus.SUCCEEDED})()


class _Checkpoints:
    def __init__(self) -> None:
        self.reads = 0

    async def get(self, execution_id: str, *, tenant_id: str) -> None:
        del execution_id, tenant_id
        self.reads += 1
        return None


class _DenyAuthorization:
    async def authorize(self, principal, action, resource) -> None:
        del principal, action, resource
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)


def test_approval_creation_is_not_a_public_runtime_operation() -> None:
    assert not hasattr(DefaultApprovalService, "create")
    assert hasattr(DefaultApprovalService, "list")
    assert hasattr(DefaultApprovalService, "decide")


@pytest.mark.asyncio
async def test_approval_list_authorizes_before_reading_deferred_state() -> None:
    checkpoints = _Checkpoints()
    service = DefaultApprovalService(
        object(),
        _Executions(),
        checkpoints,
        _DenyAuthorization(),
        objects=object(),
    )

    with pytest.raises(AIError) as error:
        await service.list(
            "execution",
            principal=Principal("caller", "tenant", "service"),
        )

    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert checkpoints.reads == 0


def test_approval_decision_request_keeps_only_external_decision_input() -> None:
    request = ApprovalDecisionRequest(
        Principal("caller", "tenant", "service"),
        "approval",
        "decision-key",
        ApprovalDecision.APPROVE,
        metadata={"source": "test"},
    )
    assert request.metadata == {"source": "test"}
    with pytest.raises(AIError) as error:
        ApprovalDecisionRequest(
            request.principal,
            "approval",
            "",
            ApprovalDecision.APPROVE,
        )
    assert error.value.code is ErrorCode.IDEMPOTENCY_KEY_INVALID
