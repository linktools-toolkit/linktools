#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution transcript paging behavior."""

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.agent import AgentBindingContract
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    agent_conversation_id as make_agent_conversation_id,
    agent_run_id as make_agent_run_id,
)
from linktools.ai.runtime._history import StepExecutionHistoryReader
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.spec import AgentSpec

from ._runtime_test_helpers import execution_owner_fields


def _binding() -> AgentBindingContract:
    output = bind_output()
    return AgentBindingContract(
        agent_spec=AgentSpec("default", model_route="default"),
        model_contract={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
    )


def _execution() -> ExecutionRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ExecutionRecord(
        execution_id="execution",
        session_id=None,
        parent_execution_id=None,
        root_execution_id="execution",
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.RUN,
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=1,
        error_code=None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
        **execution_owner_fields(),
    )


class _Executions:
    def __init__(self, record: ExecutionRecord) -> None:
        self._record = record
        self.tenant_id = "tenant"

    async def get(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord | None:
        if (
            execution_id != self._record.execution_id
            or tenant_id != self.tenant_id
        ):
            return None
        return self._record

    async def list_children(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ExecutionRecord, ...]:
        assert execution_id == self._record.execution_id
        assert tenant_id == self.tenant_id
        return ()


class _RangedStore:
    def __init__(self, agent_run_id: str) -> None:
        self._run_id = agent_run_id
        self._messages = (
            ModelRequest(parts=[UserPromptPart(content="user-1")]),
            ModelResponse(parts=[TextPart(content="assistant-1")]),
            ModelRequest(parts=[UserPromptPart(content="user-2")]),
            ModelResponse(parts=[TextPart(content="assistant-2")]),
        )
        self.ranges: list[tuple[int, int]] = []

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
        if agent_run_id != self._run_id:
            return None
        return AgentRunRecord(
            agent_run_id=agent_run_id,
            agent_conversation_id=make_agent_conversation_id(
                namespace="history",
                tenant_id="tenant",
                execution_id="execution",
            ),
            parent_agent_run_id=None,
            agent_name="default",
            metadata={"agent_run_sequence": "1", "agent_name": "default"},
            started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

    async def transcript_message_count(self, owner_id: str) -> int:
        assert owner_id == self._run_id
        return len(self._messages)

    def iter_message_range(
        self,
        *,
        agent_run_id: str,
        start: int,
        end: int,
    ) -> AsyncIterator[object]:
        assert agent_run_id == self._run_id
        self.ranges.append((start, end))

        async def iterate() -> AsyncIterator[object]:
            for message in self._messages[start:end]:
                yield message

        return iterate()


@pytest.mark.asyncio
async def test_execution_transcript_cursor_keeps_first_page_high_water() -> None:
    record = _execution()
    agent_run_id = make_agent_run_id(
        namespace="history",
        tenant_id="tenant",
        execution_id=record.execution_id,
        agent_run_sequence=1,
    )
    store = _RangedStore(agent_run_id)
    reader = StepExecutionHistoryReader(
        namespace="history",
        executions=_Executions(record),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        cursor_signer=HmacCursorSigner("history", b"history-key"),
    )

    first = await reader.transcript(
        record.execution_id,
        tenant_id="tenant",
        cursor=None,
        limit=2,
    )
    assert first.next_cursor is not None

    store._messages = (
        *store._messages,
        ModelRequest(parts=[UserPromptPart(content="late-user")]),
        ModelResponse(parts=[TextPart(content="late-assistant")]),
    )
    second = await reader.transcript(
        record.execution_id,
        tenant_id="tenant",
        cursor=first.next_cursor,
        limit=10,
    )

    assert tuple(item.text for item in second.items) == ("user-2", "assistant-2")
    assert second.next_cursor is None


@pytest.mark.asyncio
async def test_execution_transcript_cursor_resumes_from_message_range() -> None:
    record = _execution()
    agent_run_id = make_agent_run_id(
        namespace="history",
        tenant_id="tenant",
        execution_id=record.execution_id,
        agent_run_sequence=1,
    )
    store = _RangedStore(agent_run_id)
    reader = StepExecutionHistoryReader(
        namespace="history",
        executions=_Executions(record),  # type: ignore[arg-type]
        store=store,  # type: ignore[arg-type]
        cursor_signer=HmacCursorSigner("history", b"history-key"),
    )

    first = await reader.transcript(
        record.execution_id,
        tenant_id="tenant",
        cursor=None,
        limit=2,
    )
    assert tuple(item.text for item in first.items) == ("user-1", "assistant-1")
    assert first.next_cursor is not None

    second = await reader.transcript(
        record.execution_id,
        tenant_id="tenant",
        cursor=first.next_cursor,
        limit=2,
    )

    assert tuple(item.text for item in second.items) == ("user-2", "assistant-2")
    assert second.next_cursor is None
    assert store.ranges[-1] == (2, 4)
