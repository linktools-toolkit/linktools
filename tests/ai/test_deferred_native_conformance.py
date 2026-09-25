#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deferred approval and ordinary checkpoint contracts."""

from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai.exceptions import ApprovalRequired
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from linktools.ai.runtime.state._step_contracts import (
    AgentRunCheckpoint,
)
from linktools.ai.runtime.state._steps import (
    StagingAgentRunStore,
)

from linktools.ai.runtime._capture import RuntimeCaptureStore
from linktools.ai.runtime._capabilities import _RuntimeAgentRunPersistence
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.workspace import (
    WorkspaceToolPermissionPolicy,
)
from ._runtime_test_helpers import tool_with_metadata


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


class _RecordingAgentRunStore(StagingAgentRunStore):
    def __init__(self) -> None:
        super().__init__()
        self.saved_checkpoints: list[AgentRunCheckpoint] = []

    async def save_checkpoint(
        self,
        checkpoint: AgentRunCheckpoint,
        *,
        execution_id: str | None = None,
    ) -> None:
        self.saved_checkpoints.append(checkpoint)
        await super().save_checkpoint(checkpoint, execution_id=execution_id)


async def _read_file(path: str) -> str:
    return path


@pytest.mark.asyncio
async def test_approval_frontier_is_persisted_as_interrupted(tmp_path: Path) -> None:
    agent_run_id = "deferred-run"
    store = _RecordingAgentRunStore()
    bridge = _Bridge()
    captured: list[int] = []
    del tmp_path
    descriptor = ManagedToolDescriptor(
        effect_owner="tool_operation",
        effect_policy="non_replay_safe",
        tool_class="filesystem.read",
    )
    boundary = RuntimeToolBoundaryToolset(
        (FunctionToolset([tool_with_metadata(_read_file, descriptor)]),),
        {
            "_read_file": descriptor
        },
        id="workspace",
            workspace_policy=WorkspaceToolPermissionPolicy(default="ask"),
        tool_operations=bridge,  # type: ignore[arg-type]
    )

    context = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id=agent_run_id,
        tool_call_id="call",
    )
    tools = await boundary.get_tools(context)
    with pytest.raises(ApprovalRequired):
        await boundary.call_tool(
            "_read_file",
            {"path": "file.txt"},
            context,
            tools["_read_file"],
        )

    assert bridge.calls == 0
    assert not captured
    assert not store.saved_checkpoints


@pytest.mark.asyncio
async def test_ordinary_completed_checkpoint_behavior_is_unchanged() -> None:
    agent_run_id = "completed-run"
    store = _RecordingAgentRunStore()
    persistence = _RuntimeAgentRunPersistence(
        capture=RuntimeCaptureStore(
            store,
            execution_id=None,
            agent_run_id=agent_run_id,
        ),
        agent_name="agent",
        agent_run_id=agent_run_id,
    )
    agent = Agent(TestModel(custom_output_text="ok"))

    result = await agent.run(
        "finish",
        run_id=agent_run_id,
        capabilities=(persistence,),
    )

    assert result.output == "ok"
    assert store.saved_checkpoints
    latest = await store.latest_checkpoint(agent_run_id=agent_run_id)
    assert latest is not None
    assert latest.state == "complete"
    assert latest == store.saved_checkpoints[-1]
