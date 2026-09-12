#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deferred frontier and native step-persistence contracts."""

from types import SimpleNamespace
from typing import Any

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._capabilities import _RuntimeStepPersistence
from linktools.ai.runtime._harness import HarnessStepStoreAdapter
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.workspace import (
    ToolPermissionRule,
    WorkspaceToolPermissionPolicy,
)
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import DeferredToolRequests, RunContext
from pydantic_ai.usage import RunUsage
from ._runtime_test_helpers import semantic_tool


class _Bridge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def begin(self, *args: object, **kwargs: object) -> Any:
        del args, kwargs
        self.calls.append("begin")
        raise AssertionError("deferred gate must run before tool admission")

    async def renew(self, decision: Any) -> Any:
        return decision

    async def complete(self, decision: Any, result: Any) -> bool:
        del decision, result
        self.calls.append("complete")
        return False

    async def fail(self, decision: Any, error: BaseException) -> bool:
        del decision, error
        self.calls.append("fail")
        return False

    async def unknown(self, decision: Any, error: BaseException) -> None:
        del decision, error
        self.calls.append("unknown")

    async def defer(self, decision: Any) -> bool:
        del decision
        self.calls.append("defer")
        return False


class _Store:
    def __init__(self) -> None:
        self.snapshots: list[object] = []

    async def save_snapshot(self, snapshot: object) -> None:
        self.snapshots.append(snapshot)

    async def append_event(self, event: object) -> None:
        del event


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


@pytest.mark.asyncio
async def test_runtime_step_persistence_marks_native_deferred_run_interrupted() -> None:
    store = _Store()
    captured: list[int] = []
    persistence = _RuntimeStepPersistence(
        store=HarnessStepStoreAdapter(store, execution_id=None),
        agent_name="agent",
        run_id="run",
        deferred_pause_sink=captured.append,
    )
    node_result = object()
    ctx = SimpleNamespace(run_step=7, conversation_id=None, messages=[])
    assert (
        await persistence.after_node_run(
            ctx,
            node=object(),
            result=node_result,  # type: ignore[arg-type]
        )
        is node_result
    )
    deferred = DeferredToolRequests(approvals=[])
    result = SimpleNamespace(output=deferred, all_messages=lambda: [])
    assert await persistence.after_run(ctx, result=result) is result  # type: ignore[arg-type]

    assert captured == [7]
    assert len(store.snapshots) == 1
    snapshot = store.snapshots[0]
    assert getattr(snapshot, "state") == "interrupted"
    assert getattr(snapshot, "step_index") == 7


@pytest.mark.asyncio
async def test_runtime_step_persistence_requires_pause_sink_for_native_deferred() -> None:
    persistence = _RuntimeStepPersistence(
        store=HarnessStepStoreAdapter(_Store(), execution_id=None),
        agent_name="agent",
        run_id="run",
    )
    persistence._last_observed_step_index = 3
    result = SimpleNamespace(output=DeferredToolRequests(approvals=[]))
    with pytest.raises(AIError) as error:
        await persistence.after_run(
            SimpleNamespace(run_step=0),
            result=result,  # type: ignore[arg-type]
        )
    assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_ask_boundary_defers_before_runtime_operation() -> None:
    bridge = _Bridge()

    async def read_file(path: str) -> str:
        return path

    descriptor = ManagedToolDescriptor(
        effect_owner="none",
        effect="none",
        tool_class="filesystem.read",
    )
    boundary = RuntimeToolBoundaryToolset(
        (FunctionToolset([semantic_tool(read_file, descriptor)]),),
        {
            "read_file": descriptor
        },
        id="workspace",
        workspace_policy=WorkspaceToolPermissionPolicy(
            (ToolPermissionRule("ask", tool_name="read_file"),)
        ),
        tool_operations=bridge,  # type: ignore[arg-type]
    )
    context = _context()
    tools = await boundary.get_tools(context)
    with pytest.raises(ApprovalRequired):
        await boundary.call_tool(
            "read_file",
            {"path": "pkg/file.txt"},
            context,
            tools["read_file"],
        )
    assert bridge.calls == []
