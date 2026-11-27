#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Session delegation can keep optional Trace on a separate SQL route."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import SessionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus, PersistenceDomain, ToolOperationStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._canonical_trace_interactions import _CapturedTurnInteractions
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _wait_for_committed
from .test_canonical_turn_subagents import _HostAuthorization, _model


@pytest.mark.asyncio
async def test_cold_split_trace_drops_locked_archive_initialization(tmp_path: Path, monkeypatch) -> None:
    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures("locked-session-trace")
    features.model("default", _model("default", model))
    features.agent("default", model="default", allow_tools=())
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    trace_path = tmp_path / "trace.db"
    storage = PersistenceStorage.from_plan(replace(
        PersistencePlan.from_default(StorageRoute.sqlite(tmp_path / "session.db")),
        trace=StorageRoute.sqlite(trace_path),
    ))
    dropped = asyncio.Event()
    prepare = _CapturedTurnInteractions.prepare_append

    async def observe_prepare(capture, archive):
        try:
            return await prepare(capture, archive)
        except AIError as error:
            if error.code is ErrorCode.STORAGE_UNAVAILABLE:
                dropped.set()
            raise

    monkeypatch.setattr(_CapturedTurnInteractions, "prepare_append", observe_prepare)
    blocker = create_async_engine(f"sqlite+aiosqlite:///{trace_path}")
    try:
        async with Persistence.open("locked-session-trace", storage=storage) as state:
            async with blocker.begin() as writer:
                await writer.exec_driver_sql("BEGIN IMMEDIATE")
                async with Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime:
                    store = state.snapshot_resources().stores[PersistenceDomain.TRACE]
                    assert store.storage_group.opened_archive is None
                    agent = runtime.agents.get()
                    principal = runtime.default_principal
                    await agent.create_session("chat", principal=principal)
                    turn = await runtime.sessions.start(
                        "chat", "Reply", agent=agent, principal=principal, idempotency_key="reply",
                    )
                    assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
                    await asyncio.wait_for(dropped.wait(), timeout=1)
                assert store.storage_group.opened_archive is None
    finally:
        await blocker.dispose()


@pytest.mark.asyncio
async def test_split_session_trace_preserves_native_leaf_without_enabling_execution_trace(tmp_path: Path) -> None:
    calls: list[str] = []
    returned_ids: list[str] = []
    child_answer = "synthetic reviewer answer"
    call_id = "one-reviewer"

    async def leaf(_messages, info):
        calls.append("leaf")
        assert "delegate_task" not in {tool.name for tool in info.function_tools}
        return ModelResponse(parts=[TextPart(child_answer)])

    async def parent(messages, info):
        calls.append("parent")
        returns = [part for part in messages[-1].parts if isinstance(part, ToolReturnPart)]
        if not returns:
            assert "delegate_task" in {tool.name for tool in info.function_tools}
            return ModelResponse(parts=[ToolCallPart(
                "delegate_task", {"subagent_id": "reviewer", "task": "Read synthetic entry"}, tool_call_id=call_id,
            )])
        [returned] = returns
        assert returned.tool_call_id == call_id and returned.outcome == "success"
        assert returned.content["status"] == "SUCCEEDED"
        assert returned.content["output"] == {"text": child_answer}
        returned_ids.append(returned.content["execution_id"])
        return ModelResponse(parts=[TextPart("review complete")])

    features = RuntimeFeatures("split-session-child-trace")
    features.model("parent", _model("parent", parent))
    features.model("child", _model("child", leaf))
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    features.agent("default", model="parent", allow_tools=(), allow_subagents=("reviewer",))
    features.agent("reviewer", model="child", allow_tools=(), allow_subagents=())
    primary = StorageRoute.sqlite(tmp_path / "session.db")
    trace = StorageRoute.sqlite(tmp_path / "trace.db")
    storage = PersistenceStorage.from_plan(replace(PersistencePlan.from_default(primary), trace=trace))
    async with (
        Persistence.open("split-session-child-trace", storage=storage) as state,
        Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime,
    ):
        stores = state.snapshot_resources().stores
        assert stores[PersistenceDomain.TRACE].storage_group is not stores[PersistenceDomain.EXECUTION].storage_group
        agent = runtime.agents.get()
        with pytest.raises(AIError) as unsupported:
            await agent.start("This is not a Session turn", idempotency_key="standalone")
        assert unsupported.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
        assert calls == []

        principal = runtime.default_principal
        await agent.create_session("chat", principal=principal)
        turn = await runtime.sessions.start(
            "chat", "Review the entry", agent=agent, principal=principal, idempotency_key="review",
        )
        result = await turn.wait_final(timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED
        assert result.view["output"] == {"text": "review complete"}
        assert calls == ["parent", "leaf", "parent"]

        turn_id = turn.key.turn_id
        [child_id] = returned_ids
        assert child_id != turn_id
        [child] = await state.execution.executions.list_children(turn_id, tenant_id=runtime.tenant_id)
        assert child.execution_id == child_id and child.status is ExecutionStatus.SUCCEEDED
        assert not child.durable_trace
        [receipt] = await state.checkpoint.tools.list_by_execution(turn_id, tenant_id=runtime.tenant_id)
        assert receipt.tool_call_id == call_id and receipt.status is ToolOperationStatus.COMPLETED
        filtered = await _wait_for_committed(
            lambda: runtime.history.turn_trace("chat", turn_id, principal=principal),
            lambda page: bool(page.items),
        )
        assert all(item.turn_id == turn_id for item in filtered.items)
