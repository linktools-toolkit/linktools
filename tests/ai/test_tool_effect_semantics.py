#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeToolBoundary effect ownership and failure contracts."""

import asyncio
from typing import Any

import pytest
from linktools.ai.core import ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool import ToolOperationDecision
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.runtime.state._contracts import ToolOperationRecord
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    ToolFailed,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from linktools.ai.workspace import WorkspaceToolPermissionPolicy
from ._runtime_test_helpers import semantic_tool


class _Bridge:
    def __init__(
        self,
        replay_safe: bool,
        *,
        cached_result: Any = None,
        has_cached_result: bool = False,
        cached_error: BaseException | None = None,
    ) -> None:
        self.decision = ToolOperationDecision(
            "operation",
            "owner",
            1,
            replay_safe,
            cached_result=cached_result,
            has_cached_result=has_cached_result,
            cached_error=cached_error,
        )
        self.calls: list[str] = []

    async def begin(self, *args: object, **kwargs: object) -> ToolOperationDecision:
        del args, kwargs
        self.calls.append("begin")
        return self.decision

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        return decision

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del decision, result
        self.calls.append("complete")
        return False

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del decision, error
        self.calls.append("fail")
        return False

    async def unknown(
        self,
        decision: ToolOperationDecision,
        error: BaseException,
    ) -> None:
        del decision, error
        self.calls.append("unknown")

    async def defer(self, decision: ToolOperationDecision) -> bool:
        del decision
        self.calls.append("defer")
        return False


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


async def _call(
    handler: Any,
    descriptor: ManagedToolDescriptor,
    *,
    bridge: _Bridge | None = None,
    workspace_policy: Any = None,
) -> tuple[Any, _Bridge | None]:
    selected_bridge = bridge
    boundary = RuntimeToolBoundaryToolset(
        (FunctionToolset([semantic_tool(handler, descriptor)]),),
        {handler.__name__: descriptor},
        id="test.boundary",
        workspace_policy=workspace_policy,
        tool_operations=selected_bridge,  # type: ignore[arg-type]
    )
    context = _context()
    tools = await boundary.get_tools(context)
    result = await boundary.call_tool(
        handler.__name__,
        {},
        context,
        tools[handler.__name__],
    )
    return result, selected_bridge


@pytest.mark.asyncio
async def test_effect_free_tool_does_not_create_tool_operation() -> None:
    async def read() -> str:
        return "ok"

    result, bridge = await _call(
        read,
        ManagedToolDescriptor(
            effect_owner="none",
            effect="none",
            tool_class="business",
        ),
    )

    assert result == "ok"
    assert bridge is None


@pytest.mark.asyncio
async def test_workspace_approval_precedes_tool_operation_admission() -> None:
    async def write() -> str:
        raise AssertionError("approval must stop the leaf")

    bridge = _Bridge(False)
    with pytest.raises(ApprovalRequired):
        await _call(
            write,
            ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="non_replay_safe",
                tool_class="filesystem.write",
            ),
            bridge=bridge,
            workspace_policy=WorkspaceToolPermissionPolicy(default="ask"),
        )
    assert bridge.calls == []


@pytest.mark.asyncio
async def test_replay_safe_known_tool_failure_is_terminalized() -> None:
    async def retry() -> None:
        raise ModelRetry("retry")

    bridge = _Bridge(True)
    with pytest.raises(ModelRetry):
        await _call(
            retry,
            ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="replay_safe",
                tool_class="business",
            ),
            bridge=bridge,
        )
    assert bridge.calls == ["begin", "fail"]


@pytest.mark.asyncio
async def test_non_replay_safe_known_tool_failure_becomes_effect_unknown() -> None:
    async def retry() -> None:
        raise ModelRetry("retry")

    bridge = _Bridge(False)
    with pytest.raises(ToolFailed, match="TOOL_EFFECT_UNKNOWN"):
        await _call(
            retry,
            ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="non_replay_safe",
                tool_class="business",
            ),
            bridge=bridge,
        )
    assert bridge.calls == ["begin", "unknown"]


@pytest.mark.asyncio
async def test_replay_safe_unhandled_failure_becomes_effect_unknown() -> None:
    async def broken() -> None:
        raise RuntimeError("unknown result")

    bridge = _Bridge(True)
    with pytest.raises(AIError) as raised:
        await _call(
            broken,
            ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="replay_safe",
                tool_class="business",
            ),
            bridge=bridge,
        )
    assert raised.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
    assert bridge.calls == ["begin", "unknown"]


@pytest.mark.asyncio
async def test_non_replay_safe_unhandled_failure_requires_effect_verification() -> None:
    async def broken() -> None:
        raise RuntimeError("unknown result")

    bridge = _Bridge(False)
    with pytest.raises(ToolFailed, match="TOOL_EFFECT_UNKNOWN"):
        await _call(
            broken,
            ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="non_replay_safe",
                tool_class="business",
            ),
            bridge=bridge,
        )
    assert bridge.calls == ["begin", "unknown"]


@pytest.mark.asyncio
async def test_replay_safe_native_deferred_call_releases_tool_operation() -> None:
    async def deferred() -> None:
        raise CallDeferred({"reason": "later"})

    bridge = _Bridge(True)
    with pytest.raises(CallDeferred):
        await _call(
            deferred,
            ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="replay_safe",
                tool_class="business",
            ),
            bridge=bridge,
        )
    assert bridge.calls == ["begin", "defer"]


@pytest.mark.asyncio
async def test_non_replay_safe_deferred_call_becomes_effect_unknown() -> None:
    async def deferred() -> None:
        raise CallDeferred({"reason": "later"})

    bridge = _Bridge(False)
    with pytest.raises(ToolFailed, match="TOOL_EFFECT_UNKNOWN"):
        await _call(
            deferred,
            ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="non_replay_safe",
                tool_class="business",
            ),
            bridge=bridge,
        )
    assert bridge.calls == ["begin", "unknown"]


@pytest.mark.asyncio
async def test_external_cancellation_keeps_cancellation_control_flow() -> None:
    async def cancelled() -> None:
        raise asyncio.CancelledError

    bridge = _Bridge(False)
    with pytest.raises(asyncio.CancelledError):
        await _call(
            cancelled,
            ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="non_replay_safe",
                tool_class="business",
            ),
            bridge=bridge,
        )
    assert bridge.calls == ["begin", "unknown"]


@pytest.mark.asyncio
async def test_cached_result_skips_raw_leaf() -> None:
    async def unexpected() -> None:
        raise AssertionError("cached operation must not invoke the leaf")

    bridge = _Bridge(True, cached_result={"cached": True}, has_cached_result=True)
    result, _ = await _call(
        unexpected,
        ManagedToolDescriptor(
            effect_owner="tool_operation",
            effect="replay_safe",
            tool_class="business",
        ),
        bridge=bridge,
    )
    assert result == {"cached": True}
    assert bridge.calls == ["begin"]


@pytest.mark.asyncio
async def test_cached_failure_skips_raw_leaf() -> None:
    async def unexpected() -> None:
        raise AssertionError("cached operation must not invoke the leaf")

    bridge = _Bridge(True, cached_error=AIError(ErrorCode.TOOL_RETRY_REQUIRED))
    with pytest.raises(AIError) as raised:
        await _call(
            unexpected,
            ManagedToolDescriptor(
                effect_owner="tool_operation",
                effect="replay_safe",
                tool_class="business",
            ),
            bridge=bridge,
        )
    assert raised.value.code is ErrorCode.TOOL_RETRY_REQUIRED
    assert bridge.calls == ["begin"]


def test_tool_descriptor_rejects_effect_without_owner() -> None:
    with pytest.raises(ValueError):
        ManagedToolDescriptor(
            effect_owner="none",
            effect="replay_safe",
            tool_class="business",
        )


def test_tool_descriptor_rejects_tool_operation_without_effect() -> None:
    with pytest.raises(ValueError):
        ManagedToolDescriptor(
            effect_owner="tool_operation",
            effect="none",
            tool_class="business",
        )


def test_error_payload_contract_uses_durable_tool_record() -> None:
    assert ToolOperationStatus.CLAIMED.value == "CLAIMED"
    assert ToolOperationRecord.__name__ == "ToolOperationRecord"
