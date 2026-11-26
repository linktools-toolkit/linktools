#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1 approval service surface and authorization regressions."""

from types import SimpleNamespace
from collections.abc import Awaitable, Callable
from typing import TypeVar

import pytest

from linktools.ai.core import (
    ApprovalDecision,
    AuthorizationAction,
    ExecutionStatus,
    Principal,
    ResourceKind,
    ResourceRef,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ApprovalDecisionRequest
from linktools.ai.runtime._approval import DefaultApprovalService


_T = TypeVar("_T")


class _ReadStateStore:
    async def read(self, callback: Callable[[object], Awaitable[_T]]) -> _T:
        return await callback(None)


class _Executions:
    state_store = _ReadStateStore()

    async def get(self, execution_id: str, *, tenant_id: str) -> object | None:
        if execution_id == "execution" and tenant_id == "tenant":
            return SimpleNamespace(
                execution_id=execution_id, principal_id="owner", parent_execution_id=None,
                status=ExecutionStatus.SUCCEEDED,
            )
        return None


class _RecordingAuthorization:
    def __init__(self, allowed: bool) -> None:
        self.allowed = allowed
        self.calls: list[tuple[Principal, AuthorizationAction, ResourceRef]] = []

    async def authorize(self, principal, action, resource) -> None:
        self.calls.append((principal, action, resource))
        if not self.allowed:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", (False, True))
async def test_approval_list_requires_authorization_even_without_pending_tools(allowed: bool) -> None:
    policy = _RecordingAuthorization(allowed)
    service = DefaultApprovalService(object(), _Executions(), policy, objects=object())
    principal = Principal("caller", "tenant", "service")

    if allowed:
        assert await service.list("execution", principal=principal) == ()
    else:
        with pytest.raises(AIError) as error:
            await service.list("execution", principal=principal)
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED

    assert policy.calls == [(principal, AuthorizationAction.APPROVAL_READ,
                            ResourceRef(ResourceKind.EXECUTION, "execution", "tenant"))]


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
