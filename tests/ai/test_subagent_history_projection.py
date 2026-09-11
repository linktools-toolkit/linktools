#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for direct Subagent execution history reads."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    step_conversation_id,
    step_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._history import StepExecutionHistoryReader
from linktools.ai.runtime.state._contracts import ExecutionRecord
from linktools.ai.runtime.state._step_contracts import RunRecord, StepEvent
from linktools.ai.spec import AgentSpec
from ._runtime_test_helpers import execution_owner_fields


def _binding() -> AgentBindingSnapshot:
    output = bind_output()
    return AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("default", model="default"),
        base_model={"route_id": "default", "model_identity": "test:model"},
        selected=(),
        subagents=(),
        output_mode=output.mode,
        output_schema=output.schema_definition,
        binding_digest="a" * 64,
    )


def _record(
    execution_id: str,
    *,
    created_at: datetime,
    parent_execution_id: str | None = None,
    root_execution_id: str | None = None,
    parent_invocation_id: str | None = None,
) -> ExecutionRecord:
    subagent = parent_execution_id is not None
    return ExecutionRecord(
        execution_id=execution_id,
        tenant_id="tenant",
        session_id=None,
        binding_digest="a" * 64,
        parent_execution_id=parent_execution_id,
        root_execution_id=root_execution_id or execution_id,
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=(
            ExecutionLineageKind.SUBAGENT if subagent else ExecutionLineageKind.RUN
        ),
        status=ExecutionStatus.STARTED,
        revision=0,
        event_sequence=0,
        agent_run_sequence=1,
        error_code=None,
        safe_error_details={},
        created_at=created_at,
        updated_at=created_at,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
        parent_invocation_id=parent_invocation_id,
        **execution_owner_fields(),
    )


class _Executions:
    def __init__(self, records: tuple[ExecutionRecord, ...]) -> None:
        self._records = {record.execution_id: record for record in records}
        self.list_children_calls: list[str] = []

    async def get(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> ExecutionRecord | None:
        assert tenant_id == "tenant"
        return self._records.get(execution_id)

    async def list_children(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> tuple[ExecutionRecord, ...]:
        assert tenant_id == "tenant"
        self.list_children_calls.append(execution_id)
        return tuple(
            record
            for record in self._records.values()
            if record.parent_execution_id == execution_id
        )


class _Store:
    def __init__(self, records: tuple[ExecutionRecord, ...]) -> None:
        self._runs: dict[str, tuple[ExecutionRecord, RunRecord]] = {}
        for record in records:
            run_id = step_run_id(
                namespace="history",
                tenant_id="tenant",
                execution_id=record.execution_id,
                segment_sequence=1,
            )
            conversation_id = step_conversation_id(
                namespace="history",
                tenant_id="tenant",
                execution_id=record.execution_id,
            )
            self._runs[run_id] = (
                record,
                RunRecord(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    agent_name="default",
                    metadata={"segment_sequence": "1"},
                ),
            )

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        value = self._runs.get(run_id)
        return None if value is None else value[1]

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        record, run = self._runs[run_id]
        return [
            StepEvent(
                run_id=run_id,
                kind="model_request_started",
                step_index=0,
                timestamp=record.created_at,
                conversation_id=run.conversation_id,
                agent_name="default",
            )
        ]

    async def iter_messages(self, *, run_id: str):
        record, _run = self._runs[run_id]
        yield ModelRequest(
            parts=[UserPromptPart(content=f"prompt:{record.execution_id}")]
        )
        yield ModelResponse(
            parts=[TextPart(content=f"response:{record.execution_id}")]
        )


def _reader(
    records: tuple[ExecutionRecord, ...],
) -> tuple[StepExecutionHistoryReader, _Executions]:
    executions = _Executions(records)
    return (
        StepExecutionHistoryReader(
            namespace="history",
            executions=executions,  # type: ignore[arg-type]
            store=_Store(records),  # type: ignore[arg-type]
            cursor_signer=HmacCursorSigner("history", b"history-key"),
        ),
        executions,
    )


@pytest.mark.asyncio
async def test_subagent_history_trace_and_transcript_read_directly() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    root = _record("root", created_at=created_at)
    child = _record(
        "child",
        created_at=created_at + timedelta(seconds=1),
        parent_execution_id="root",
        root_execution_id="root",
        parent_invocation_id="delegate-call",
    )
    reader, executions = _reader((root, child))

    history = await reader.history("child", tenant_id="tenant", cursor=None, limit=100)
    trace = await reader.trace("child", tenant_id="tenant", cursor=None, limit=100)
    transcript = await reader.transcript(
        "child", tenant_id="tenant", cursor=None, limit=100
    )

    assert history.items
    assert {item.execution_id for item in history.items} == {"child"}
    assert trace.items
    assert {item.execution_id for item in trace.items} == {"child"}
    assert trace.items[0].payload["scope"] == "subagent"
    assert trace.items[0].payload["depth"] == 1
    assert trace.items[0].payload["child_execution_id"] == "child"
    assert transcript.items
    assert {item.execution_id for item in transcript.items} == {"child"}
    assert executions.list_children_calls == []


@pytest.mark.asyncio
async def test_root_history_projects_direct_subagent_once() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    root = _record("root", created_at=created_at)
    child = _record(
        "child",
        created_at=created_at + timedelta(seconds=1),
        parent_execution_id="root",
        root_execution_id="root",
        parent_invocation_id="delegate-call",
    )
    reader, executions = _reader((root, child))

    history = await reader.history("root", tenant_id="tenant", cursor=None, limit=100)

    assert {item.execution_id for item in history.items} == {"root", "child"}
    assert executions.list_children_calls == ["root"]


@pytest.mark.asyncio
async def test_root_history_rejects_inconsistent_direct_subagent_lineage() -> None:
    created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    root = _record("root", created_at=created_at)
    child = _record(
        "child",
        created_at=created_at + timedelta(seconds=1),
        parent_execution_id="root",
        root_execution_id="other-root",
        parent_invocation_id="delegate-call",
    )
    reader, _executions = _reader((root, child))

    with pytest.raises(AIError) as raised:
        await reader.history("root", tenant_id="tenant", cursor=None, limit=100)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
