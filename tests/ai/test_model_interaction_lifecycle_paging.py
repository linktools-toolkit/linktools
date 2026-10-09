#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model interaction paging keeps one fixed request identity set."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    agent_conversation_id,
    agent_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime._model_interaction import (
    StagedContextProjection,
    StagedModelInteraction,
)
from linktools.ai.runtime.service_api import UsageReadCutoff
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    ModelInteractionRecord,
    RuntimePayloadRef,
)
from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import AgentRunHistoryCapture, AgentRunRecord, StepEvent
from linktools.ai.storage import StoredPayload


class _Executions:
    def __init__(self, root: object) -> None:
        self.root = root
        self.children: list[object] = []

    async def get(self, execution_id: str, *, tenant_id: str) -> object | None:
        del tenant_id
        if execution_id == "root":
            return self.root
        return next(
            (
                child
                for child in self.children
                if child.execution_id == execution_id
            ),
            None,
        )

    async def list_children(
        self,
        execution_id: str,
        *,
        tenant_id: str,
    ) -> list[object]:
        del tenant_id
        return [
            child
            for child in self.children
            if child.parent_execution_id == execution_id
        ]


class _HistoryStore:
    def __init__(self) -> None:
        self.runs: dict[str, AgentRunRecord] = {}
        self.interactions: dict[str, list[ModelInteractionRecord]] = {}
        self.before_reads: dict[str, list[ModelInteractionRecord]] = {}

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
        return self.runs.get(agent_run_id)

    async def list_events(self, *, agent_run_id: str) -> list[StepEvent]:
        del agent_run_id
        return []

    async def model_interaction_count(self, *, agent_run_id: str) -> int:
        return max((value.model_request_seq for value in self.interactions.get(agent_run_id, ())), default=0)

    async def capture_history(self, agent_run_ids, *, include_pending=False):
        del include_pending
        result = {}
        for run_id in agent_run_ids:
            run = await self.get_agent_run(agent_run_id=run_id)
            count = await self.model_interaction_count(agent_run_id=run_id)
            if run is None and count:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            result[run_id] = AgentRunHistoryCapture(run, 0, 0, count)
        return result

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[ModelInteractionRecord]:
        for updated in self.before_reads.pop(agent_run_id, ()):
            self.interactions[agent_run_id] = [
                updated if value.model_request_seq == updated.model_request_seq else value
                for value in self.interactions[agent_run_id]
            ]
        values = [
            item
            for item in self.interactions.get(agent_run_id, ())
            if after_model_request_seq is None
            or item.model_request_seq > after_model_request_seq
        ]
        return values if limit is None else values[:limit]

    async def resolve_model_interactions(
        self,
        interactions: Any,
    ) -> list[object]:
        del interactions
        raise AssertionError("lightweight queries do not resolve request content")


def _record(execution_id: str, *, child: bool = False) -> object:
    now = datetime.now(timezone.utc)
    return type(
        "Execution",
        (),
        {
            "execution_id": execution_id,
            "parent_execution_id": "root" if child else None,
            "root_execution_id": "root",
            "lineage_kind": (
                ExecutionLineageKind.SUBAGENT
                if child
                else ExecutionLineageKind.RUN
            ),
            "parent_invocation_id": "call-1" if child else None,
            "status": ExecutionStatus.STARTED,
            "agent_run_seq": 1,
            "created_at": now + (timedelta(seconds=1) if child else timedelta()),
        },
    )()


def _interaction(agent_id: str, sequence: int) -> ModelInteractionRecord:
    started_at = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(
        milliseconds=sequence
    )
    return ModelInteractionRecord(
        agent_id,
        step_index=sequence,
        model_request_seq=sequence,
        purpose="agent",
        output_retry_index=None,
        model={"route_id": "default"},
        request_context=ContextProjection(()),
        request_envelope=RuntimePayloadRef(
            StoredPayload.inline_text("{}"),
            RuntimeDomain.EXECUTION,
        ),
        response_context=None,
        status="CANCELLED",
        error_code=None,
        duration_ns=0,
        usage=None,
        started_at=started_at,
        finished_at=started_at + timedelta(milliseconds=1),
    )


def _running_interaction(
    execution_id: str,
    run_sequence: int,
    model_request_seq: int,
    started_at: datetime,
) -> StagedModelInteraction:
    run_id = agent_run_id(
        namespace="history",
        tenant_id="tenant",
        execution_id=execution_id,
        agent_run_seq=run_sequence,
    )
    return StagedModelInteraction(
        agent_run_id=run_id,
        step_index=model_request_seq,
        model_request_seq=model_request_seq,
        purpose="agent",
        output_retry_index=None,
        model={"route_id": "default"},
        request_context=StagedContextProjection(()),
        request_envelope_digest="0" * 64,
        response_context=None,
        status="RUNNING",
        error_code=None,
        duration_ns=None,
        usage=None,
        started_at=started_at,
        finished_at=None,
    )


def _running_record(execution_id: str, run_sequence: int, sequence: int, started_at: datetime) -> ModelInteractionRecord:
    run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id=execution_id, agent_run_seq=run_sequence)
    return replace(_interaction(run_id, sequence), status="RUNNING", started_at=started_at,
                   duration_ns=None, finished_at=None, request_context=None, request_envelope=None)


def _terminal(value: ModelInteractionRecord | StagedModelInteraction) -> ModelInteractionRecord | StagedModelInteraction:
    return replace(
        value,
        status="CANCELLED",
        duration_ns=14,
        finished_at=value.started_at + timedelta(milliseconds=1)
        if value.started_at is not None
        else None,
    )


def _reader() -> tuple[StepExecutionHistoryReader, _Executions, _HistoryStore]:
    root = _record("root")
    executions = _Executions(root)
    store = _HistoryStore()
    conv_id = agent_conversation_id(
        namespace="history",
        tenant_id="tenant",
        execution_id="root",
    )
    run_id = agent_run_id(
        namespace="history",
        tenant_id="tenant",
        execution_id="root",
        agent_run_seq=1,
    )
    store.runs[run_id] = AgentRunRecord(
        run_id,
        agent_conversation_id=conv_id,
        agent_id="agent",
        metadata={"agent_run_seq": "1"},
    )
    store.interactions[run_id] = [
        _running_record(
            "root",
            1,
            sequence,
            datetime.now(timezone.utc) + timedelta(milliseconds=sequence),
        )
        for sequence in (1, 2)
    ]
    return (
        StepExecutionHistoryReader(
            namespace="history",
            executions=executions,  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            cursor_signer=HmacCursorSigner("lifecycle", b"lifecycle-key"),
        ),
        executions,
        store,
    )


@pytest.mark.asyncio
async def test_terminal_update_after_capture_preserves_request_identity() -> None:
    reader, _executions, store = _reader()
    run_id = next(iter(store.interactions))
    store.before_reads[run_id] = [_interaction(run_id, 1)]

    page = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=None,
        limit=1,
        include_content=False,
    )

    assert [(item.model_request_seq, item.status) for item in page.items] == [
        (1, "CANCELLED")
    ]


@pytest.mark.asyncio
async def test_pages_refresh_states_without_expanding_captured_identity_set() -> None:
    reader, executions, store = _reader()
    first = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=None,
        limit=1,
        include_content=False,
    )
    assert [(item.execution_id, item.model_request_seq, item.status) for item in first.items] == [
        ("root", 1, "RUNNING")
    ]
    assert first.next_cursor is not None

    root_run_id = next(iter(store.interactions))
    store.interactions[root_run_id] = [
        *(_terminal(value) for value in store.interactions[root_run_id]),
        _running_record("root", 1, 3, datetime.now(timezone.utc)),
    ]
    child = _record("child", child=True)
    executions.children.append(child)
    child_conv_id = agent_conversation_id(
        namespace="history",
        tenant_id="tenant",
        execution_id="child",
    )
    child_run_id = agent_run_id(
        namespace="history",
        tenant_id="tenant",
        execution_id="child",
        agent_run_seq=1,
    )
    store.runs[child_run_id] = AgentRunRecord(
        child_run_id,
        agent_conversation_id=child_conv_id,
        agent_id="child-agent",
        metadata={"agent_run_seq": "1"},
    )
    store.interactions[child_run_id] = [
        _running_record("child", 1, 1, datetime.now(timezone.utc)),
    ]

    second = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=first.next_cursor,
        limit=10,
        include_content=False,
    )
    assert [(item.execution_id, item.model_request_seq, item.status) for item in second.items] == [
        ("root", 2, "CANCELLED")
    ]

    refreshed = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=None,
        limit=10,
        include_content=False,
    )
    assert [
        (item.execution_id, item.model_request_seq, item.status)
        for item in refreshed.items
    ] == [
        ("root", 1, "CANCELLED"),
        ("root", 2, "CANCELLED"),
        ("root", 3, "RUNNING"),
        ("child", 1, "RUNNING"),
    ]


@pytest.mark.asyncio
async def test_explicit_cutoffs_use_the_same_lifecycle_paging_contract() -> None:
    reader, _executions, store = _reader()
    run_id = next(iter(store.interactions))
    store.interactions[run_id] = [
        _interaction(run_id, 1),
        _interaction(run_id, 2),
    ]

    assert (
        await reader.model_interactions(
            "root",
            tenant_id="tenant",
            cursor=None,
            limit=10,
            cutoffs=(),
            include_content=False,
        )
    ).items == ()

    cutoff = (UsageReadCutoff("root", 1, 2),)
    page = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=None,
        limit=1,
        cutoffs=cutoff,
        include_content=False,
    )
    assert [(item.model_request_seq, item.status) for item in page.items] == [
        (1, "CANCELLED")
    ]
    assert page.next_cursor is not None

    omitted = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=page.next_cursor,
        limit=1,
        include_content=False,
    )
    same = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=page.next_cursor,
        limit=1,
        cutoffs=cutoff,
        include_content=False,
    )
    assert [(item.model_request_seq, item.status) for item in omitted.items] == [
        (2, "CANCELLED")
    ]
    assert omitted.items == same.items

    with pytest.raises(AIError) as changed:
        await reader.model_interactions(
            "root",
            tenant_id="tenant",
            cursor=page.next_cursor,
            limit=1,
            cutoffs=(UsageReadCutoff("root", 1, 1),),
            include_content=False,
        )
    assert changed.value.code is ErrorCode.CURSOR_INVALID
