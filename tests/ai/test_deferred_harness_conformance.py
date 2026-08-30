#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolved Harness conformance for deferred and ordinary snapshots."""

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    InMemoryStepStore,
)

from linktools.ai.runtime._agent_executor import AgentExecutor, _RuntimePersistenceBoundary
from linktools.ai.runtime._capabilities import _RuntimeStepPersistence, _WorkspaceToolGate
from linktools.ai.workspace import (
    RepositoryInstructions,
    ToolPermissionRule,
    WorkspacePolicy,
    WorkspaceToolPermissionPolicy,
)


class _Bridge:
    def __init__(self) -> None:
        self.calls = 0

    async def begin(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        raise AssertionError("deferred gate must run before tool-operation admission")

    async def renew(self, decision):
        return decision

    async def complete(self, decision, result):
        del decision, result
        return False

    async def fail(self, decision, error):
        del decision, error
        return False

    async def unknown(self, decision, error) -> None:
        del decision, error

    async def existing_call_ids(self, tool_call_ids):
        del tool_call_ids
        return frozenset()


class _RecordingStepStore(InMemoryStepStore):
    def __init__(self) -> None:
        super().__init__()
        self.saved_snapshots: list[ContinuableSnapshot] = []

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        self.saved_snapshots.append(snapshot)
        await super().save_snapshot(snapshot)


class _EmptyResolver:
    async def resolve(
        self,
        path: str | Path = ".",
        *,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> RepositoryInstructions:
        del path, exclude_sources
        return RepositoryInstructions(())


async def _read_file(path: str) -> str:
    return path


@pytest.mark.asyncio
async def test_resolved_harness_never_persists_open_approval_frontier(tmp_path: Path) -> None:
    run_id = "deferred-harness-run"
    store = _RecordingStepStore()
    bridge = _Bridge()
    captured: list[int] = []
    persistence = _RuntimeStepPersistence(
        tool_operations=bridge,
        store=store,
        agent_name="agent",
        run_id=run_id,
        trusted_tool_classes=(("_read_file", "filesystem.read"),),
        deferred_pause_sink=captured.append,
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
                (ToolPermissionRule("ask", tool_name="_read_file"),)
            )
        ),
        trusted_tool_classes=(("_read_file", "filesystem.read"),),
    )
    agent = Agent(
        TestModel(call_tools=["_read_file"]),
        tools=[_read_file],
        output_type=[str, DeferredToolRequests],
    )

    result = await agent.run(
        "read it",
        run_id=run_id,
        capabilities=(gate, _RuntimePersistenceBoundary(persistence)),
    )

    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.approvals
    assert not result.output.calls
    assert captured and len(captured) == 1 and captured[0] > 0
    assert bridge.calls == 0
    for snapshot in store.saved_snapshots:
        assert AgentExecutor.pending_tool_calls(snapshot.messages, run_id=run_id) == ()


@pytest.mark.asyncio
async def test_resolved_harness_ordinary_completed_snapshot_behavior_is_unchanged() -> None:
    run_id = "completed-harness-run"
    store = _RecordingStepStore()
    persistence = _RuntimeStepPersistence(
        tool_operations=_Bridge(),
        store=store,
        agent_name="agent",
        run_id=run_id,
    )
    agent = Agent(TestModel(custom_output_text="ok"))

    result = await agent.run(
        "finish",
        run_id=run_id,
        capabilities=(_RuntimePersistenceBoundary(persistence),),
    )

    assert result.output == "ok"
    assert store.saved_snapshots
    latest = await store.latest_snapshot(run_id=run_id)
    assert latest is not None
    assert latest.state == "complete"
    assert latest == store.saved_snapshots[-1]
