#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.core import (
    ExecutionEventType,
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
)
from linktools.ai.runtime._execution_tree import (
    ExecutionTreeBroker,
    ExecutionTreeStreamer,
)
from linktools.ai.runtime.service_api import (
    ExecutionStreamEvent,
    ExecutionView,
)
from linktools.ai.runtime.state import ExecutionRecord
from linktools.ai.spec import AgentSpec


def _binding(agent_id: str = "agent") -> AgentBindingSnapshot:
    spec = AgentSpec(agent_id)
    return AgentBindingSnapshot(
        version=1,
        agent_spec=spec,
        model={},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={},
        binding_digest="a" * 64,
    )


def _record(*, subagent: bool, parent_invocation_id: str | None) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id="child" if subagent else "root",
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id="root" if subagent else None,
        root_execution_id="root",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=(
            ExecutionLineageKind.SUBAGENT
            if subagent
            else ExecutionLineageKind.RUN
        ),
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=0,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
        parent_invocation_id=parent_invocation_id,
    )


def test_subagent_lineage_requires_parent_invocation() -> None:
    with pytest.raises(ValueError):
        _record(subagent=True, parent_invocation_id=None)
    child = _record(subagent=True, parent_invocation_id="tool-call")
    assert child.parent_invocation_id == "tool-call"
    with pytest.raises(ValueError):
        _record(subagent=False, parent_invocation_id="tool-call")


class _ExecutionReader:
    def __init__(self) -> None:
        self.root = ExecutionView(
            "root",
            "root-agent",
            ExecutionStatus.STARTED,
            ExecutionLineageKind.RUN,
            None,
            "root",
            None,
        )
        self.child = ExecutionView(
            "child",
            "child-agent",
            ExecutionStatus.SUCCEEDED,
            ExecutionLineageKind.SUBAGENT,
            "root",
            "root",
            "delegate-call",
        )

    async def inspect(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> ExecutionView:
        del principal
        return self.root if execution_id == "root" else self.child

    async def list_children(
        self,
        execution_id: str,
        *,
        principal: Principal,
    ) -> tuple[ExecutionView, ...]:
        del principal
        return (self.child,) if execution_id == "root" else ()


class _EventStreamer:
    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ):
        del principal

        async def events():
            sequence = after_sequence + 1
            yield ExecutionStreamEvent(
                execution_id,
                sequence,
                ExecutionEventType.EXECUTION_SUCCEEDED,
                {},
            )

        return events()


@pytest.mark.asyncio
async def test_tree_stream_projects_root_and_child_without_global_sequence() -> None:
    broker = ExecutionTreeBroker()
    streamer = ExecutionTreeStreamer(
        _ExecutionReader(),
        _EventStreamer(),
        broker,
    )
    values = [
        item
        async for item in streamer.stream(
            "root",
            principal=Principal("owner", "tenant", "service"),
            after_sequences={"child": 4},
        )
    ]
    assert {(item.execution_id, item.depth) for item in values} == {
        ("root", 0),
        ("child", 1),
    }
    child = next(
        item for item in values if item.execution_id == "child"
    )
    assert child.parent_invocation_id == "delegate-call"
    assert child.event.durable_sequence == 5
