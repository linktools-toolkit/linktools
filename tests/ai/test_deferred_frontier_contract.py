#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deferred human-approval frontier contracts."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests, RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.step_persistence import StepPersistence

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_executor import AgentExecutor, _RuntimePersistenceBoundary
from linktools.ai.runtime._capabilities import ToolOperationDecision, _RuntimeStepPersistence
from linktools.ai.runtime._repository_instructions import _WorkspaceToolGate
from linktools.ai.workspace import (
    RepositoryInstructions,
    ToolPermissionRule,
    WorkspacePolicy,
    WorkspaceToolPermissionPolicy,
)


class _Bridge:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def begin(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args, replay_safe
        self.calls.append("begin")
        return ToolOperationDecision("operation", "owner", 1, True)

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

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None:
        del decision, error
        self.calls.append("unknown")

    async def existing_call_ids(self, tool_call_ids: tuple[str, ...]) -> frozenset[str]:
        del tool_call_ids
        return frozenset()


class _Store:
    def __init__(self) -> None:
        self.snapshots: list[object] = []
        self.effects: list[object] = []

    async def save_snapshot(self, snapshot: object) -> None:
        self.snapshots.append(snapshot)

    async def record_tool_effect(self, effect: object) -> None:
        self.effects.append(effect)


class _EmptyResolver:
    async def resolve(
        self,
        target: str,
        *,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> RepositoryInstructions:
        del target, exclude_sources
        return RepositoryInstructions(())


def _context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id="run")


def _approval_call(call_id: str = "approval-1") -> ToolCallPart:
    return ToolCallPart(
        tool_name="read_file",
        args={"path": "pkg/file.txt"},
        tool_call_id=call_id,
    )


@pytest.mark.asyncio
async def test_runtime_step_persistence_uses_last_observed_step_and_sink_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _Bridge()
    store = _Store()
    captured: list[int] = []
    persistence = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
        deferred_pause_sink=captured.append,
    )
    node_result = object()
    after_node = AsyncMock(return_value=node_result)
    after_run = AsyncMock(side_effect=lambda _ctx, *, result: result)
    monkeypatch.setattr(StepPersistence, "after_node_run", after_node)
    monkeypatch.setattr(StepPersistence, "after_run", after_run)

    ctx = SimpleNamespace(run_step=7)
    assert await persistence.after_node_run(
        ctx, node=object(), result=node_result  # type: ignore[arg-type]
    ) is node_result
    ctx.run_step = 0
    deferred = DeferredToolRequests(approvals=[_approval_call()])
    result = SimpleNamespace(output=deferred)
    assert await persistence.after_run(ctx, result=result) is result  # type: ignore[arg-type]

    assert captured == [7]
    assert store.snapshots == []
    assert bridge.calls == []
    after_node.assert_awaited_once()
    after_run.assert_awaited_once()


@pytest.mark.asyncio
async def test_runtime_step_persistence_rejects_generic_deferred_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = _RuntimeStepPersistence(
        tool_operations=_Bridge(),
        store=_Store(),
        agent_name="agent",
        run_id="run",
        deferred_pause_sink=lambda _step: None,
    )
    monkeypatch.setattr(
        StepPersistence,
        "after_run",
        AsyncMock(side_effect=lambda _ctx, *, result: result),
    )
    persistence._last_observed_step_index = 3
    result = SimpleNamespace(output=DeferredToolRequests(calls=[_approval_call("external")]))
    with pytest.raises(AIError) as error:
        await persistence.after_run(SimpleNamespace(run_step=0), result=result)  # type: ignore[arg-type]
    assert error.value.code is ErrorCode.CAPABILITY_POLICY_CONFLICT


@pytest.mark.asyncio
async def test_ask_gate_defers_before_runtime_operation_or_harness_effect(tmp_path) -> None:
    bridge = _Bridge()
    store = _Store()
    persistence = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id="run",
        trusted_tool_classes=(("read_file", "filesystem.read"),),
    )
    gate = _WorkspaceToolGate(
        execution_id="execution",
        workspace_root=tmp_path,
        repository_instruction_history=(),
        repository_instruction_marker_authority=frozenset(),
        repository_instructions=RepositoryInstructions(()),
        instruction_resolver=_EmptyResolver(),
        policy=WorkspacePolicy(
            tool_permissions=WorkspaceToolPermissionPolicy(
                (ToolPermissionRule("ask", tool_name="read_file"),)
            )
        ),
        trusted_tool_classes=(("read_file", "filesystem.read"),),
    )
    combined = CombinedCapability((_RuntimePersistenceBoundary(persistence), gate))
    call = _approval_call()
    definition = ToolDefinition(
        name="read_file",
        capability_id="workspace-filesystem",
        metadata={"linktools.ai.replay_safe": True},
    )
    with pytest.raises(ApprovalRequired):
        await combined.before_tool_execute(
            _context(),
            call=call,
            tool_def=definition,
            args={"path": "pkg/file.txt"},
        )
    assert bridge.calls == []
    assert store.effects == []
    assert store.snapshots == []
    assert not persistence._calls


def test_pending_tool_calls_keeps_only_unresolved_current_run_in_model_order() -> None:
    approval_a = ToolCallPart("approve_a", {"x": 1}, tool_call_id="a")
    terminal = ToolCallPart("settled", {"y": 2}, tool_call_id="settled")
    approval_b = ToolCallPart("approve_b", {"z": 3}, tool_call_id="b")
    messages = (
        ModelResponse(parts=[approval_a, terminal, approval_b], run_id="run"),
        ModelRequest(
            parts=[ToolReturnPart("settled", "ok", tool_call_id="settled")],
            run_id="run",
        ),
        ModelResponse(
            parts=[ToolCallPart("foreign", {}, tool_call_id="foreign")],
            run_id="other-run",
        ),
    )
    pending = AgentExecutor.pending_tool_calls(messages, run_id="run")
    assert pending == (approval_a, approval_b)


def test_pending_tool_calls_rejects_orphan_terminal_and_duplicate_call_id() -> None:
    with pytest.raises(AIError) as orphan:
        AgentExecutor.pending_tool_calls(
            (
                ModelRequest(
                    parts=[ToolReturnPart("tool", "ok", tool_call_id="missing")],
                    run_id="run",
                ),
            ),
            run_id="run",
        )
    assert orphan.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    call = ToolCallPart("tool", {}, tool_call_id="same")
    with pytest.raises(AIError) as duplicate:
        AgentExecutor.pending_tool_calls(
            (ModelResponse(parts=[call, call], run_id="run"),),
            run_id="run",
        )
    assert duplicate.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
