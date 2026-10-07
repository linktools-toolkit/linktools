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
from linktools.ai.runtime.state._step_contracts import AgentRunRecord, StepEvent
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
        after_model_request_seq: int | None = None,
        limit: int | None = None,
    ) -> list[ModelInteractionRecord]:
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


class _StagingStore(_HistoryStore):
    def __init__(self) -> None:
        super().__init__()
        self.staged: dict[str, dict[int, StagedModelInteraction]] = {}
        self.handoff_during_snapshot: dict[str, list[ModelInteractionRecord]] = {}

    async def model_interaction_history_high_water(
        self,
        *,
        agent_run_id: str,
    ) -> int:
        archived = await self.list_model_interactions(agent_run_id=agent_run_id)
        return max(
            max((item.model_request_seq for item in archived), default=0),
            max(self.staged.get(agent_run_id, {}), default=0),
        )

    async def list_model_interaction_history_snapshot(
        self,
        *,
        agent_run_id: str,
        after_model_request_seq: int,
        limit: int,
    ) -> tuple[list[object], list[StagedModelInteraction]]:
        staged = [
            item
            for sequence, item in sorted(self.staged.get(agent_run_id, {}).items())
            if sequence > after_model_request_seq
        ][:limit]
        for interaction in self.handoff_during_snapshot.pop(agent_run_id, ()):
            self.interactions.setdefault(agent_run_id, []).append(interaction)
            self.staged.get(agent_run_id, {}).pop(interaction.model_request_seq, None)
        archived = await self.list_model_interactions(
            agent_run_id=agent_run_id,
            after_model_request_seq=after_model_request_seq,
            limit=limit,
        )
        return archived, staged


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


def _terminal(value: StagedModelInteraction) -> StagedModelInteraction:
    return replace(
        value,
        status="CANCELLED",
        duration_ns=14,
        finished_at=value.started_at + timedelta(milliseconds=1)
        if value.started_at is not None
        else None,
    )


def _reader() -> tuple[StepExecutionHistoryReader, _Executions, _StagingStore]:
    root = _record("root")
    executions = _Executions(root)
    store = _StagingStore()
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
    store.interactions[run_id] = []
    store.staged[run_id] = {
        sequence: _running_interaction(
            "root",
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
            staging_store=store,
        ),
        executions,
        store,
    )


@pytest.mark.asyncio
async def test_archive_terminal_wins_when_handoff_follows_staging_snapshot() -> None:
    reader, _executions, store = _reader()
    run_id = next(iter(store.staged))
    store.handoff_during_snapshot[run_id] = [_interaction(run_id, 1)]

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

    root_run_id = next(iter(store.staged))
    store.staged[root_run_id][1] = _terminal(store.staged[root_run_id][1])
    store.staged[root_run_id][2] = _terminal(store.staged[root_run_id][2])
    store.staged[root_run_id][3] = _running_interaction(
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
        agent_run_seq=1,
    )
    store.runs[child_run_id] = AgentRunRecord(
        child_run_id,
        agent_conversation_id=child_conv_id,
        agent_id="child-agent",
        metadata={"agent_run_seq": "1"},
    )
    store.interactions[child_run_id] = []
    store.staged[child_run_id] = {
        1: _running_interaction("child", 1, 1, datetime.now(timezone.utc))
    }

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
    run_id = next(iter(store.staged))
    store.interactions[run_id] = [
        _interaction(run_id, 1),
        _interaction(run_id, 2),
    ]
    store.staged[run_id].clear()

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
