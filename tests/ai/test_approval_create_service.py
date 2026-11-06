#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V1 approval service surface and authorization regressions."""

from types import SimpleNamespace
from collections.abc import Awaitable, Callable
from typing import TypeVar

import pytest

from linktools.ai.core import (
    ApprovalDecision,
    ExecutionStatus,
    Principal,
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
                execution_id=execution_id, principal_id="owner",
                status=ExecutionStatus.SUCCEEDED,
            )
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
async def test_approval_list_requires_authorization_even_without_pending_tools() -> None:
    service = DefaultApprovalService(
        object(),
        _Executions(),
        _DenyAuthorization(),
        objects=object(),
    )

    with pytest.raises(AIError) as error:
        await service.list(
            "execution",
            principal=Principal("caller", "tenant", "service"),
        )

    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


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
