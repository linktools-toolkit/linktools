#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model interaction cursors bind lifecycle and usage query semantics."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    UsageMetrics,
    agent_conversation_id,
    agent_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime._model_interaction import ModelInteractionLifecycle
from linktools.ai.runtime.service_api import UsageReadCutoff
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    ModelInteractionRecord,
    RuntimePayloadRef,
)
from linktools.ai.runtime.state._plan import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import AgentRunRecord, StepEvent
from linktools.ai.storage import StoredPayload


class _Executions:
    def __init__(self, root: object) -> None:
        self.root = root
        self.children: list[object] = []

    async def get(self, execution_id: str, *, tenant_id: str) -> object | None:
        del tenant_id
        return self.root if execution_id == "root" else None

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

    async def get_agent_run(self, *, agent_run_id: str) -> AgentRunRecord | None:
        return self.runs.get(agent_run_id)

    async def list_events(self, *, agent_run_id: str) -> list[StepEvent]:
        del agent_run_id
        return []

    async def model_interaction_count(self, *, agent_run_id: str) -> int:
        return len(self.interactions.get(agent_run_id, ()))

    async def list_model_interactions(
        self,
        *,
        agent_run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[ModelInteractionRecord]:
        values = [
            item
            for item in self.interactions.get(agent_run_id, ())
            if after_request_sequence is None
            or item.request_sequence > after_request_sequence
        ]
        return values if limit is None else values[:limit]

    async def resolve_model_interactions(
        self,
        interactions: tuple[ModelInteractionRecord, ...],
    ) -> list[object]:
        del interactions
        raise AssertionError("lightweight queries do not resolve request content")


class _LifecycleStore(_HistoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.live: dict[str, dict[int, ModelInteractionLifecycle]] = {}
        self.handoff_during_snapshot: dict[str, list[ModelInteractionRecord]] = {}

    async def model_interaction_history_high_water(
        self,
        *,
        agent_run_id: str,
    ) -> int:
        archived = await self.list_model_interactions(agent_run_id=agent_run_id)
        return max(
            max((item.request_sequence for item in archived), default=0),
            max(self.live.get(agent_run_id, {}), default=0),
        )

    async def list_model_interaction_lifecycle(
        self,
        *,
        agent_run_id: str,
        after_request_sequence: int | None = None,
        limit: int | None = None,
    ) -> list[ModelInteractionLifecycle]:
        values = [
            item
            for sequence, item in sorted(self.live.get(agent_run_id, {}).items())
            if after_request_sequence is None or sequence > after_request_sequence
        ]
        return values if limit is None else values[:limit]

    async def list_model_interaction_history_snapshot(
        self,
        *,
        agent_run_id: str,
        after_request_sequence: int,
        limit: int,
    ) -> tuple[list[object], list[ModelInteractionLifecycle]]:
        lifecycle = await self.list_model_interaction_lifecycle(
            agent_run_id=agent_run_id,
            after_request_sequence=after_request_sequence,
            limit=limit,
        )
        for interaction in self.handoff_during_snapshot.pop(agent_run_id, ()):
            self.interactions.setdefault(agent_run_id, []).append(interaction)
            self.live.get(agent_run_id, {}).pop(interaction.request_sequence, None)
        archived = await self.list_model_interactions(
            agent_run_id=agent_run_id,
            after_request_sequence=after_request_sequence,
            limit=limit,
        )
        return archived, lifecycle


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
            "agent_run_sequence": 1,
            "created_at": now + (timedelta(seconds=1) if child else timedelta()),
        },
    )()


def _interaction(agent_id: str, sequence: int) -> ModelInteractionRecord:
    return ModelInteractionRecord(
        agent_id,
        step_index=sequence,
        request_sequence=sequence,
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
    )


def _running_interaction(
    execution_id: str,
    run_sequence: int,
    request_sequence: int,
    started_at: datetime,
) -> ModelInteractionLifecycle:
    run_id = agent_run_id(
        namespace="history",
        tenant_id="tenant",
        execution_id=execution_id,
        agent_run_sequence=run_sequence,
    )
    return ModelInteractionLifecycle(
        agent_run_id=run_id,
        step_index=request_sequence,
        request_sequence=request_sequence,
        purpose="agent",
        output_retry_index=None,
        model={"route_id": "default"},
        request={"messages": [{"text": "private"}]},
        response=None,
        status="RUNNING",
        error_code=None,
        duration_ns=None,
        usage=None,
        started_at=started_at,
    )


def _terminal(value: ModelInteractionLifecycle) -> ModelInteractionLifecycle:
    return ModelInteractionLifecycle(
        agent_run_id=value.agent_run_id,
        step_index=value.step_index,
        request_sequence=value.request_sequence,
        purpose=value.purpose,
        output_retry_index=value.output_retry_index,
        model=value.model,
        request=value.request,
        response=None,
        status="CANCELLED",
        error_code=None,
        duration_ns=14,
        usage=None,
        started_at=value.started_at,
        finished_at=value.started_at + timedelta(milliseconds=1),
    )


def _reader() -> tuple[StepExecutionHistoryReader, _Executions, _LifecycleStore]:
    root = _record("root")
    executions = _Executions(root)
    store = _LifecycleStore()
    for execution_id in ("root",):
        conv_id = agent_conversation_id(
            namespace="history",
            tenant_id="tenant",
            execution_id=execution_id,
        )
        run_id = agent_run_id(
            namespace="history",
            tenant_id="tenant",
            execution_id=execution_id,
            agent_run_sequence=1,
        )
        store.runs[run_id] = AgentRunRecord(
            run_id,
            agent_conversation_id=conv_id,
            agent_id="agent",
            metadata={"agent_run_sequence": "1"},
        )
        store.interactions[run_id] = []
        store.live[run_id] = {
            sequence: _running_interaction(
                execution_id,
                1,
                sequence,
                datetime.now(timezone.utc) + timedelta(milliseconds=sequence),
            )
            for sequence in (1, 2)
        }
    return (
        StepExecutionHistoryReader(
            namespace="history",
            executions=executions,  # type: ignore[arg-type]
            store=store,  # type: ignore[arg-type]
            cursor_signer=HmacCursorSigner("lifecycle", b"lifecycle-key"),
            lifecycle_store=store,
        ),
        executions,
        store,
    )


@pytest.mark.asyncio
async def test_archive_terminal_wins_when_handoff_follows_lifecycle_snapshot() -> None:
    reader, _executions, store = _reader()
    run_id = next(iter(store.live))
    store.handoff_during_snapshot[run_id] = [_interaction(run_id, 1)]

    page = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=None,
        limit=1,
        include_content=False,
    )

    assert [(item.request_sequence, item.status) for item in page.items] == [
        (1, "CANCELLED")
    ]


@pytest.mark.asyncio
async def test_lifecycle_pages_refresh_unreturned_states_without_expanding_identity_set() -> None:
    reader, executions, store = _reader()
    first = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=None,
        limit=1,
        include_content=False,
    )
    assert [(item.execution_id, item.request_sequence, item.status) for item in first.items] == [
        ("root", 1, "RUNNING")
    ]
    assert first.next_cursor is not None
    assert first.items[0].request == {}
    assert first.items[0].started_at is not None
    assert first.items[0].finished_at is None
    assert first.items[0].usage is None

    root_run_id = next(iter(store.live))
    store.live[root_run_id][1] = _terminal(store.live[root_run_id][1])
    store.live[root_run_id][2] = _terminal(store.live[root_run_id][2])
    store.live[root_run_id][3] = _running_interaction(
        "root", 1, 3, datetime.now(timezone.utc)
    )
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
        agent_run_sequence=1,
    )
    store.runs[child_run_id] = AgentRunRecord(
        child_run_id,
        agent_conversation_id=child_conv_id,
        agent_id="child-agent",
        metadata={"agent_run_sequence": "1"},
    )
    store.interactions[child_run_id] = []
    store.live[child_run_id] = {
        1: _running_interaction("child", 1, 1, datetime.now(timezone.utc))
    }

    second = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=first.next_cursor,
        limit=10,
        include_content=False,
    )
    assert [(item.execution_id, item.request_sequence, item.status) for item in second.items] == [
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
        (item.execution_id, item.request_sequence, item.status)
        for item in refreshed.items
    ] == [
        ("root", 1, "CANCELLED"),
        ("root", 2, "CANCELLED"),
        ("root", 3, "RUNNING"),
        ("child", 1, "RUNNING"),
    ]


@pytest.mark.asyncio
async def test_usage_cutoff_pagination_modes_are_bound_to_the_cursor() -> None:
    reader, _executions, store = _reader()
    run_id = next(iter(store.live))
    store.interactions[run_id] = [
        _interaction(run_id, 1),
        _interaction(run_id, 2),
    ]
    store.live[run_id].clear()
    assert (await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=None,
        limit=10,
        cutoffs=(),
        include_content=False,
    )).items == ()

    cutoff = (UsageReadCutoff("root", 1, 2),)
    usage_page = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=None,
        limit=1,
        cutoffs=cutoff,
        include_content=False,
    )
    assert [(item.request_sequence, item.status) for item in usage_page.items] == [
        (1, "CANCELLED")
    ]
    assert usage_page.next_cursor is not None
    omitted = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=usage_page.next_cursor,
        limit=1,
        include_content=False,
    )
    same = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=usage_page.next_cursor,
        limit=1,
        cutoffs=cutoff,
        include_content=False,
    )
    assert [(item.request_sequence, item.status) for item in omitted.items] == [
        (2, "CANCELLED")
    ]
    assert omitted.items == same.items
    with pytest.raises(AIError) as changed:
        await reader.model_interactions(
            "root",
            tenant_id="tenant",
            cursor=usage_page.next_cursor,
            limit=1,
            cutoffs=(UsageReadCutoff("root", 1, 1),),
            include_content=False,
        )
    assert changed.value.code is ErrorCode.CURSOR_INVALID

    lifecycle_page = await reader.model_interactions(
        "root",
        tenant_id="tenant",
        cursor=None,
        limit=1,
        include_content=False,
    )
    assert lifecycle_page.next_cursor is not None
    with pytest.raises(AIError) as changed_mode:
        await reader.model_interactions(
            "root",
            tenant_id="tenant",
            cursor=lifecycle_page.next_cursor,
            limit=1,
            cutoffs=(),
            include_content=False,
        )
    assert changed_mode.value.code is ErrorCode.CURSOR_INVALID
